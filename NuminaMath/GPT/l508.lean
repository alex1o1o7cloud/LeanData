import Mathlib
import Mathlib.Algebra
import Mathlib.Algebra.Factorial
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Order.Floor
import Mathlib.Algebra.Polynomial.Basic
import Mathlib.Algebra.Quadratic
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Calculus.FDeriv
import Mathlib.Analysis.Geometry.Circle
import Mathlib.Analysis.Geometry.Euclidean.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Complex
import Mathlib.Analysis.Trigonometry.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.Data.GCD
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.List.Permutation
import Mathlib.Data.List.Range
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Polynomial.Degree.Definitions
import Mathlib.Data.Rat.Defs
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Geometry.Triangle.Basic
import Mathlib.Init.Data.Int.Basic
import Mathlib.MeasureTheory.Geometry
import Mathlib.MeasureTheory.MeasurableSpace
import Mathlib.Probability.Independence
import Mathlib.Probability.Sum
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.LibrarySearch
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Solver
import Mathlib.Topology.Algebra.Polynomial
import Real
import data.rat.basic

namespace tangent_line_at_2_8_l508_508479

theorem tangent_line_at_2_8 (a : ℝ) : (∀ x y : ℝ, (y = x^3) → (x = 2) → (y = 8) → 12 * x - a * y - 16 = 0) → a = 1 :=
by
  assume h,
  have h1 : 12 * 2 - a * 8 - 16 = 0,
  from h 2 8 (by norm_num) rfl rfl,
  sorry

end tangent_line_at_2_8_l508_508479


namespace scientific_notation_correct_l508_508920

theorem scientific_notation_correct (n : ℕ) (h : n = 11580000) : n = 1.158 * 10^7 := 
sorry

end scientific_notation_correct_l508_508920


namespace pentagon_area_l508_508272

open Function 

/-
Given a convex pentagon FGHIJ with the following properties:
  1. ∠F = ∠G = 100°
  2. JF = FG = GH = 3
  3. HI = IJ = 5
Prove that the area of pentagon FGHIJ is approximately 15.2562 square units.
-/

noncomputable def area_pentagon_FGHIJ : ℝ :=
  let sin100 := Real.sin (100 * Real.pi / 180)
  let area_FGJ := (3 * 3 * sin100) / 2
  let area_HIJ := (5 * 5 * Real.sqrt 3) / 4
  area_FGJ + area_HIJ

theorem pentagon_area : abs (area_pentagon_FGHIJ - 15.2562) < 0.0001 := by
  sorry

end pentagon_area_l508_508272


namespace area_of_quadrilateral_EFCD_l508_508947

def trapezoid_area (AB CD altitude : ℝ) (EF : ℝ) : ℝ :=
  let h_EFCD := (2 / 3) * altitude
  let base_sum := EF + CD
  (h_EFCD * base_sum) / 2

theorem area_of_quadrilateral_EFCD (AB CD altitude : ℝ) (E F : ℝ) 
  (h1 : AB = 10)
  (h2 : CD = 25)
  (h3 : altitude = 15)
  (h4 : EF = (1 / 3) * AB + (2 / 3) * CD) :
  trapezoid_area AB CD altitude EF = 225 := 
by
  sorry

end area_of_quadrilateral_EFCD_l508_508947


namespace value_of_f_at_7_l508_508120

def f (x : ℝ) : ℝ := (2 * x + 3) / (4 * x - 5)

theorem value_of_f_at_7 : f 7 = 17 / 23 := by
  sorry

end value_of_f_at_7_l508_508120


namespace poly_in_An_mul_l508_508981

-- Define the conditions for a polynomial to belong to A(n)
def in_An (n : ℕ) (P : Polynomial ℝ) : Prop :=
  let coeffs := P.coeffs in
  (0 ≤ coeffs 0) ∧ (coeffs 0 = coeffs n) ∧
  (∀ i, 1 ≤ i ∧ i ≤ n/2 → coeffs i = coeffs (n - i)) ∧
  (∀ i, 0 ≤ i ∧ i < n/2 → coeffs i ≤ coeffs (i + 1))

-- Main theorem statement
theorem poly_in_An_mul (P : Polynomial ℝ) (Q : Polynomial ℝ) (n m : ℕ)
  (hP : in_An n P) (hQ : in_An m Q) :
  in_An (n + m) (P * Q) :=
sorry

end poly_in_An_mul_l508_508981


namespace scientific_notation_11580000_l508_508914

theorem scientific_notation_11580000 :
  (11580000 : ℝ) = 1.158 * 10^7 :=
sorry

end scientific_notation_11580000_l508_508914


namespace vector_angle_l508_508873

open Real

variables (a b : ℝ^3)

def magnitude (v : ℝ^3) := sqrt (v.dot v)

def angle (u v : ℝ^3) := acos ((u.dot v) / (magnitude u * magnitude v))

theorem vector_angle
  (h1 : magnitude a = 1)
  (h2 : magnitude b = 1)
  (h3 : magnitude (a + b) = 1) :
  angle a b = 2 * π / 3 :=
by
  sorry

end vector_angle_l508_508873


namespace economic_model_l508_508362

theorem economic_model :
  ∃ (Q_s : ℝ → ℝ) (T t_max T_max : ℝ),
  (∀ P : ℝ, Q_d P = 688 - 4 * P) ∧
  (∀ P_e Q_e : ℝ, 1.5 * (4 * P_e / Q_e) = (Q_s'.eval P_e / Q_e)) ∧
  (Q_s 64 = 72) ∧
  (∀ P : ℝ, Q_s P = 6 * P - 312) ∧
  (T = 6480) ∧
  (t_max = 60) ∧
  (T_max = 8640)
where 
  Q_d: ℝ → ℝ := λ P, 688 - 4 * P
  Q_s'.eval : ℝ → ℝ := sorry

end economic_model_l508_508362


namespace parcel_post_cost_l508_508275

def indicator (P : ℕ) : ℕ := if P >= 5 then 1 else 0

theorem parcel_post_cost (P : ℕ) : 
  P ≥ 0 →
  (C : ℕ) = 15 + 5 * (P - 1) - 8 * indicator P :=
sorry

end parcel_post_cost_l508_508275


namespace rounding_to_two_decimal_places_l508_508735

theorem rounding_to_two_decimal_places (x : ℝ) (h : x = 2.7982) : 
  Real.approx x 0.01 = 2.80 :=
by 
  rw h
  exact Real.approximation_rounding_method 2.7982 0.01
  sorry

end rounding_to_two_decimal_places_l508_508735


namespace sum_of_first_twelve_terms_l508_508431

section ArithmeticSequence

variables (a : ℕ → ℚ) (d : ℚ) (a₁ : ℚ)

-- General definition of the nth term in arithmetic progression
def arithmetic_term (n : ℕ) : ℚ := a₁ + (n - 1) * d

-- Given conditions in the problem
axiom fifth_term : arithmetic_term a₁ d 5 = 1
axiom seventeenth_term : arithmetic_term a₁ d 17 = 18

-- Define the sum of the first n terms in arithmetic sequence
def sum_arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  n * (2 * a₁ + (n - 1) * d) / 2

-- Statement of the proof problem
theorem sum_of_first_twelve_terms : 
  sum_arithmetic_sequence a₁ d 12 = 37.5 := 
sorry

end ArithmeticSequence

end sum_of_first_twelve_terms_l508_508431


namespace bus_speed_excluding_stoppages_l508_508416

-- Define the conditions
def speed_including_stoppages : ℝ := 48 -- km/hr
def stoppage_time_per_hour : ℝ := 15 / 60 -- 15 minutes is 15/60 hours

-- The main theorem stating what we need to prove
theorem bus_speed_excluding_stoppages : ∃ v : ℝ, (v * (1 - stoppage_time_per_hour) = speed_including_stoppages) ∧ v = 64 :=
begin
  sorry,
end

end bus_speed_excluding_stoppages_l508_508416


namespace cost_of_one_cone_l508_508595

variable (cost_of_two_cones : ℕ)
variable (h : cost_of_two_cones = 198)

theorem cost_of_one_cone (cost_of_two_cones : ℕ) (h : cost_of_two_cones = 198) : cost_of_two_cones / 2 = 99 := by
  rwa [h]
  sorry

end cost_of_one_cone_l508_508595


namespace min_PA_add_PO_l508_508111

noncomputable def F : ℝ × ℝ := (2, 0) -- focus of the parabola y^2 = -8x at (2, 0)
def O : ℝ × ℝ := (0, 0) -- origin
def P (y : ℝ) : ℝ × ℝ := (2, y) -- P is a point on the directrix x = 2
def A : ℝ × ℝ := (-2, 4) -- A is the point on the parabola y^2 = -8x with |AF| = 4

theorem min_PA_add_PO : 
  ∀ (p : ℝ × ℝ), p ∈ {p : ℝ × ℝ | p.1 = 2} → 
  |dist p A + dist p O| ≥ 2 * real.sqrt 13 :=
  sorry

end min_PA_add_PO_l508_508111


namespace value_of_a_2009_value_of_a_2014_l508_508818

noncomputable def a : ℕ → ℕ
| (4 * n - 3) := 1
| (4 * n - 1) := 0
| (2 * n) := a n
| n := sorry  -- handle other cases appropriately

theorem value_of_a_2009 : a 2009 = 1 :=
by sorry

theorem value_of_a_2014 : a 2014 = 0 :=
by sorry

end value_of_a_2009_value_of_a_2014_l508_508818


namespace slope_of_line_l508_508002

-- Define the circle and the point
def circle (x y : ℝ) : Prop := x^2 + y^2 = 16
def point_on_plane (x y : ℝ) : Prop := true  -- A trivial predicate for points on the plane
def origin : ℝ × ℝ := (0, 0)
def O : ℝ × ℝ := origin
def P (x y : ℝ) : Prop := circle x y

-- Define the slope of a line passing through (0, sqrt(5))
def line_through_point (k : ℝ) : ℝ × ℝ → Prop := λ p, p.2 = k * p.1 + (sqrt 5)

-- Define the condition vectors
def vector_eq (A B : ℝ × ℝ) : ℝ × ℝ := (A.1 + B.1, A.2 + B.2)

-- Define the initial conditions
def initial_conditions (x y : ℝ) (k : ℝ) : Prop :=
  (∃ A B : ℝ × ℝ, circle A.1 A.2 ∧ circle B.1 B.2 ∧
  P x y ∧ vector_eq O (vector_eq A B) = (x, y) ∧ line_through_point k ((0:ℝ), sqrt 5))

-- The main theorem to prove
theorem slope_of_line (k : ℝ) (x y : ℝ) (hx : initial_conditions x y k) : k = 1/2 ∨ k = -1/2 :=
sorry

end slope_of_line_l508_508002


namespace find_AM_length_in_triangle_l508_508823

noncomputable def AM_length_equilateral_triangle 
  (a : ℝ) 
  (h_pos : a > 0) 
  (BC : ℝ := a) 
  (M : ℝ → ℝ → ℝ := λ BM CM, BM = 2 * CM ∧ BM + CM = BC) : ℝ :=
  let AM_sq := a^2 + (a / 3)^2 - 2 * a * (a / 3) * (1 / 2) in
  sqrt (7 / 9 * a^2)

theorem find_AM_length_in_triangle 
  (a : ℝ) 
  (h_pos : a > 0) 
  (h_AM_eq : AM_length_equilateral_triangle a h_pos = a * sqrt 7 / 3) : 
  AM_length_equilateral_triangle a h_pos = a * sqrt 7 / 3 :=
by
  sorry

end find_AM_length_in_triangle_l508_508823


namespace largest_value_l508_508664

theorem largest_value :
  let A := 3 + 1 + 4
  let B := 3 * 1 + 4
  let C := 3 + 1 * 4
  let D := 3 * 1 * 4
  let E := 3 + 0 * 1 + 4
  D > A ∧ D > B ∧ D > C ∧ D > E :=
by
  -- conditions
  let A := 3 + 1 + 4
  let B := 3 * 1 + 4
  let C := 3 + 1 * 4
  let D := 3 * 1 * 4
  let E := 3 + 0 * 1 + 4
  -- sorry to skip the proof
  sorry

end largest_value_l508_508664


namespace area_of_any_triangle_l508_508266

theorem area_of_any_triangle 
  (h1 : ∀ (b : ℝ) (h : ℝ) (A : Type) [Triangle A], acute A → area A = 0.5 * b * h)
  (h2 : ∀ (b : ℝ) (h : ℝ) (A : Type) [Triangle A], right A → area A = 0.5 * b * h)
  (h3 : ∀ (b : ℝ) (h : ℝ) (A: Type) [Triangle A], obtuse A → area A = 0.5 * b * h)
  (b : ℝ) (h : ℝ) (A : Type) [Triangle A] :
  area A = 0.5 * b * h := 
  sorry

end area_of_any_triangle_l508_508266


namespace number_of_balls_sample_space_game_fairness_l508_508548

-- Define the initial problem conditions
def conditions (P_red_or_yellow P_yellow_or_blue : ℚ) : Prop :=
  P_red_or_yellow = 3 / 4 ∧ P_yellow_or_blue = 1 / 2

-- Prove the number of red, yellow, and blue balls
theorem number_of_balls (P_red P_yellow P_blue : ℚ) (h : conditions (P_red + P_yellow) (P_yellow + P_blue)) :
  let total_balls := 4 in
  let n_red := total_balls * P_red in
  let n_yellow := total_balls * P_yellow in
  let n_blue := total_balls * P_blue in
  n_red = 2 ∧ n_yellow = 1 ∧ n_blue = 1 := 
sory

-- Prove the sample space
theorem sample_space :
  let Ω := {(1, 1), (1, 2), (1, 'a'), (1, 'b'), (2, 1), (2, 2), (2, 'a'), (2, 'b'), ('a', 1), ('a', 2),
            ('a', 'a'), ('a', 'b'), ('b', 1), ('b', 2), ('b', 'a'), ('b', 'b')} in
  (∃ P_red P_yellow P_blue : ℚ, conditions (P_red + P_yellow) (P_yellow + P_blue)) := 
sory

-- Prove the game fairness
theorem game_fairness :
  let Ω := {(1, 1), (1, 2), (1, 'a'), (1, 'b'), (2, 1), (2, 2), (2, 'a'), (2, 'b'), ('a', 1), ('a', 2),
            ('a', 'a'), ('a', 'b'), ('b', 1), ('b', 2), ('b', 'a'), ('b', 'b')} in
  let M := {(1, 1), (2, 2), ('a', 'a'), ('b', 'b')} in
  let P_M := (M.card : ℚ) / (Ω.card : ℚ) in
  let P_N := 1 - P_M in
  P_N > P_M := 
sory

end number_of_balls_sample_space_game_fairness_l508_508548


namespace neg_mul_reverses_inequality_l508_508094

theorem neg_mul_reverses_inequality (a b : ℝ) (h : a < b) : -3 * a > -3 * b :=
  sorry

end neg_mul_reverses_inequality_l508_508094


namespace quadrilateral_similar_ratio_l508_508252

-- Define the problem context with given conditions
theorem quadrilateral_similar_ratio
    (ABCD : Quadrilateral)
    (right_angle_at_B : is_right_angle ABCD B)
    (right_angle_at_C : is_right_angle ABCD C)
    (ABC_sim_BCD : Similarity (triangle A B C) (triangle B C D))
    (AB_gt_BC : AB > BC)
    (exists_point_E : ∃ E, E ∈ Interior ABCD ∧ Similarity (triangle A B C) (triangle C E B) ∧ area (triangle A E D) = 13 * area (triangle C E B)) :
  AB / BC = 3 + 2 * sqrt 2 :=
sorry

end quadrilateral_similar_ratio_l508_508252


namespace isosceles_triangle_probability_on_cube_l508_508870

theorem isosceles_triangle_probability_on_cube : 
  let vertices := 8
  let combinations := Nat.choose vertices 3
  let isos_tris_on_faces := 4 * 6
  let isos_tris_other := 8
  let total_isos_tris := isos_tris_on_faces + isos_tris_other
  let probability := total_isos_tris / combinations
  in probability = 4 / 7 := 
by
  sorry

end isosceles_triangle_probability_on_cube_l508_508870


namespace permutationsWithSum4_l508_508506

noncomputable def numPermutationsWithSum4 : Nat :=
  -- Number of permutations (a₁, a₂, ..., a₁₀) such that ∑ |aᵢ - i| = 4
  (List.permutations [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).count (λ a => (List.sum (a.zipWith (λ ai i => (Int.natAbs (ai - i))) [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])) = 4)

theorem permutationsWithSum4 : numPermutationsWithSum4 = 52 := sorry

end permutationsWithSum4_l508_508506


namespace smallest_b_for_quadratic_factorization_l508_508800

theorem smallest_b_for_quadratic_factorization : ∃ (b : ℕ), 
  (∀ r s : ℤ, (r * s = 4032) ∧ (r + s = b) → b ≥ 127) ∧ 
  (∃ r s : ℤ, (r * s = 4032) ∧ (r + s = b) ∧ (b = 127))
:= sorry

end smallest_b_for_quadratic_factorization_l508_508800


namespace acute_triangle_exist_equal_segments_l508_508998

theorem acute_triangle_exist_equal_segments {A B C A_1 B_1 C_1 : Type} 
  (triangleABC : Triangle A B C)
  (on_sides : A_1 ∈ line B C ∧ B_1 ∈ line C A ∧ C_1 ∈ line A B) : 
  (acute_triangle A B C) ↔ (∃ x, AA_1 = x ∧ BB_1 = x ∧ CC_1 = x) := 
sorry

end acute_triangle_exist_equal_segments_l508_508998


namespace find_z_solutions_l508_508683

open Real

noncomputable def is_solution (z : ℝ) : Prop :=
  sin z + sin (2 * z) + sin (3 * z) = cos z + cos (2 * z) + cos (3 * z)

theorem find_z_solutions (z : ℝ) : 
  (∃ k : ℤ, z = 2 * π / 3 * (3 * k - 1)) ∨ 
  (∃ k : ℤ, z = 2 * π / 3 * (3 * k + 1)) ∨ 
  (∃ k : ℤ, z = π / 8 * (4 * k + 1)) ↔
  is_solution z :=
by
  sorry

end find_z_solutions_l508_508683


namespace transformed_graph_eq_l508_508674

-- Definitions based on conditions
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Points lying on the original function
def points_on_f (a b c : ℝ) : Prop := 
  f a b c 0 = 1 ∧ 
  f a b c 1 = -2 ∧ 
  f a b c (-1) = 2

-- The new function based on the transformed coefficients
def g (a b c : ℝ) (x : ℝ) : ℝ := c * x^2 + 2 * b * x + a

-- Statement equating the new function to the derived quadratic expression
theorem transformed_graph_eq (a b c : ℝ) 
  (h1 : points_on_f a b c) : g a b c = (λ x, x^2 - 4 * x - 1) :=
by
  sorry

end transformed_graph_eq_l508_508674


namespace no_common_terms_except_one_l508_508213

-- Define the sequences {x_n} and {y_n}
def x : ℕ → ℕ
| 0       := 1
| 1       := 1
| (n + 1) := x n + 2 * x (n - 1)

def y : ℕ → ℕ
| 0       := 1
| 1       := 7
| (n + 1) := 2 * y n + 3 * y (n - 1)

-- The theorem statement
theorem no_common_terms_except_one :
  ∀ n m, x n = y m → x n = 1 :=
by
  sorry

end no_common_terms_except_one_l508_508213


namespace find_a1_l508_508863

open Nat

theorem find_a1 (a : ℕ → ℕ) (h1 : ∀ n : ℕ, 0 < n → a (n + 1) = 2 * a n)
  (h2 : a 3 = 12) : a 1 = 3 :=
sorry

end find_a1_l508_508863


namespace normal_number_inequality_l508_508456

theorem normal_number_inequality (b : ℕ → ℝ) (hpos : ∀ n, 0 < b n) :
  (∃ k > 0, ∀ n, (∑ j in Finset.range n, b j) < k * b (n + 1)) ↔
  (∃ c > 0, ∃ r > 0, ∀ n, (b (n + 1) > c * b n) ∧ (b (n + r) > 2 * b n)) :=
sorry

end normal_number_inequality_l508_508456


namespace time_to_Lake_Park_restaurant_l508_508549

def time_to_Hidden_Lake := 15
def time_back_to_Park_Office := 7
def total_time_gone := 32

theorem time_to_Lake_Park_restaurant : 
  (total_time_gone = time_to_Hidden_Lake + time_back_to_Park_Office +
  (32 - (time_to_Hidden_Lake + time_back_to_Park_Office))) -> 
  (32 - (time_to_Hidden_Lake + time_back_to_Park_Office) = 10) := by
  intros 
  sorry

end time_to_Lake_Park_restaurant_l508_508549


namespace range_of_a_l508_508168

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a * x + Real.log x - (x^2 / (x - Real.log x))

theorem range_of_a (a : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ f a x1 = 0 ∧ f a x2 = 0 ∧ f a x3 = 0) ↔
  1 < a ∧ a < (Real.exp 1) / (Real.exp 1 - 1) - 1 / Real.exp 1 :=
sorry

end range_of_a_l508_508168


namespace range_of_y_l508_508476

-- Given function f with range (0, 8)
variable {α : Type*} (f : α → ℝ) (x : α) 

-- Define the condition for f(x)
def f_range_condition : Prop := 0 < f(x) ∧ f(x) ≤ 8

-- Define the quadratic function y
def y (t : ℝ) : ℝ := t^2 - 10*t - 4

-- Theorem statement
theorem range_of_y : f_range_condition f x → ∃ y, y = (f x)^2 - 10 * (f x) - 4 ∧ -29 ≤ y ∧ y < -4 := 
by 
  sorry

end range_of_y_l508_508476


namespace f_f_neg2_eq_neg2_l508_508976

noncomputable def f (x : ℝ) : ℝ := 
  if x > 0 then Real.log10 x else 10^x

theorem f_f_neg2_eq_neg2 : f (f (-2)) = -2 := by
  sorry

end f_f_neg2_eq_neg2_l508_508976


namespace cos_sin_sum_zero_l508_508401

theorem cos_sin_sum_zero : 
  cos (-11 * π / 6) + sin (11 * π / 3) = 0 := 
by 
  sorry

end cos_sin_sum_zero_l508_508401


namespace evaluated_sum_approx_2024_l508_508061

def numerator := (∑ i in Finset.range (2023 + 1), (2024 - i) / i)
def denominator := (∑ i in Finset.range (2024 - 1), 1 / (i + 2))

theorem evaluated_sum_approx_2024 :
  (numerator / denominator) = 2024 - (1 / denominator) :=
by { sorry }

end evaluated_sum_approx_2024_l508_508061


namespace max_area_cross_section_rect_prism_l508_508371

/-- The maximum area of the cross-sectional cut of a rectangular prism 
having its vertical edges parallel to the z-axis, with cross-section 
rectangle of sides 8 and 12, whose bottom side lies in the xy-plane 
centered at the origin (0,0,0), cut by the plane 3x + 5y - 2z = 30 
is approximately 118.34. --/
theorem max_area_cross_section_rect_prism :
  ∃ A : ℝ, abs (A - 118.34) < 0.01 :=
sorry

end max_area_cross_section_rect_prism_l508_508371


namespace pyramid_volume_correct_slant_height_PM_correct_l508_508605

-- Given definitions for the problem setup
def AB := 10
def BC := 5
def PA := 7
def base_area := AB * BC

-- Given that PA is perpendicular to both AB and AD, and the height from the apex to the base is PA
def pyramid_volume : ℚ := (1 / 3) * base_area * PA

-- Calculate the midpoint M of AB
def midpoint_M := (AB / 2, 0)

-- Calculate the slant height PM using the distance formula
def slant_height_PM : ℝ := real.sqrt ((midpoint_M.1^2) + (PA^2))

-- Lean 4 statement to verify the calculated values
theorem pyramid_volume_correct : pyramid_volume = 350 / 3 := by
  sorry

theorem slant_height_PM_correct : slant_height_PM = real.sqrt 74 := by
  sorry

end pyramid_volume_correct_slant_height_PM_correct_l508_508605


namespace quadrilateral_pyramid_properties_l508_508014

noncomputable def quadrilateral_pyramid : Prop :=
  ∃ (S A B C D : Type) (O H : Type),
    -- Conditions
    inscribed_in_sphere S A B C D ∧
    center_in_plane_of_base O A B C D ∧
    diagonals_intersect_at H A C B D ∧
    height_of_pyramid S H A B C D ∧
    -- Segment lengths
    CH = 4 ∧
    AS = 3^(3/4) ∧
    AD = 3 ∧
    AB = BS ∧
    -- Conclusion
    CS = 5 ∧
    CD = 16 / 3

theorem quadrilateral_pyramid_properties : quadrilateral_pyramid :=
by
  -- The detailed geometric proof goes here.
  sorry

end quadrilateral_pyramid_properties_l508_508014


namespace constant_term_l508_508824

theorem constant_term (n : ℕ) (h : (Nat.choose n 4 * 2^4) / (Nat.choose n 2 * 2^2) = (56 / 3)) :
  (∃ k : ℕ, k = 2 ∧ n = 10 ∧ Nat.choose 10 k * 2^k = 180) := by
  sorry

end constant_term_l508_508824


namespace range_of_a_l508_508776

theorem range_of_a (a : ℝ) : 
  (∃ x y : ℝ, (x - a)^2 + (y - a)^2 = 8 ∧ x^2 + y^2 = 2) ↔ 
  a ∈ set.Ioc (-3) (-1) ∪ set.Ioc 1 3 :=
by sorry

end range_of_a_l508_508776


namespace floor_ceil_product_l508_508409

theorem floor_ceil_product :
  let f : ℕ → ℤ := λ n, (Int.floor (- (n : ℤ) - 0.5)) * (Int.ceil ((n : ℤ) + 0.5))
  let product : ℤ := ∏ i in Finset.range 7, -(i + 1)^2
  ∑ n in Finset.range 7, f n = product :=
by
  sorry

end floor_ceil_product_l508_508409


namespace Gamma_area_correct_l508_508449

open Complex

noncomputable def Gamma_area : ℂ → ℝ :=
  λ z, if 2 ≤ Complex.abs z ∧ Complex.abs z < 5 then
         Real.pi * (5^2) - Real.pi * (2^2)
       else 0

theorem Gamma_area_correct (z : ℂ) (hz : 2 ≤ Complex.abs z ∧ Complex.abs z < 5) :
  Gamma_area z = 21 * Real.pi := by
  sorry

end Gamma_area_correct_l508_508449


namespace max_complexity_51_l508_508590

-- Define the complexity of a number 
def complexity (x : ℚ) : ℕ := sorry -- Placeholder for the actual complexity function definition

-- Define the sequence for m values
def m_sequence (k : ℕ) : List ℕ :=
  List.range' 1 (2^(k-1)) |>.filter (λ n => n % 2 = 1)

-- Define the candidate number
def candidate_number (k : ℕ) : ℚ :=
  (2^(k + 1) + (-1)^k) / (3 * 2^k)

theorem max_complexity_51 : 
  ∃ m, m ∈ m_sequence 50 ∧ 
  (∀ n, n ∈ m_sequence 50 → complexity (n / 2^50) ≤ complexity (candidate_number 50 / 2^50)) :=
sorry

end max_complexity_51_l508_508590


namespace patricia_earns_more_than_jose_l508_508959

noncomputable def jose_final_amount : ℝ :=
  50000 * (1 + 0.04)^2

noncomputable def patricia_final_amount : ℝ :=
  50000 * (1 + 0.01)^8

theorem patricia_earns_more_than_jose :
  patricia_final_amount - jose_final_amount = 63 :=
by
  -- from solution steps
  /-
  jose_final_amount = 50000 * (1 + 0.04)^2 = 54080
  patricia_final_amount = 50000 * (1 + 0.01)^8 ≈ 54143
  patricia_final_amount - jose_final_amount ≈ 63
  -/
  sorry

end patricia_earns_more_than_jose_l508_508959


namespace largest_valid_student_number_l508_508879

-- Define the valid two-digit multiples of 23
def is_valid_multiple_of_23 (n : ℕ) : Prop :=
  n = 23 ∨ n = 46 ∨ n = 69 ∨ n = 92

-- Check if a number formed by two adjacent digits is divisible by 23
def adjacent_digits_form_valid_multiple (x y : ℕ) : Prop :=
  is_valid_multiple_of_23 (10 * x + y)

-- Define a sequence property where any two adjacent digits must form a valid multiple of 23
def valid_student_number (s : List ℕ) : Prop :=
  match s with
  | [] => false
  | x :: xs => List.Pairwise adjacent_digits_form_valid_multiple (x :: xs)
  
-- The largest number satisfying the given constraints
def largest_student_number := 46923

-- Prove that 46923 is the largest valid student number
theorem largest_valid_student_number :
  ∃ s, valid_student_number (Int.digits 10 largest_student_number) ∧ 
  ∀ s', valid_student_number s' → (Int.of_digits 10 s' ≤ largest_student_number) := by
  sorry

end largest_valid_student_number_l508_508879


namespace bob_weekly_profit_l508_508755

-- Definitions for the conditions mentioned
def cost_per_muffin := 0.75
def selling_price_per_muffin := 1.5
def muffins_per_day := 12
def days_per_week := 7

-- Define the theorem to state the weekly profit calculation
theorem bob_weekly_profit : 
  let profit_per_muffin := selling_price_per_muffin - cost_per_muffin
  let total_daily_profit := profit_per_muffin * muffins_per_day
  let weekly_profit := total_daily_profit * days_per_week
  weekly_profit = 63 := 
by 
  sorry


end bob_weekly_profit_l508_508755


namespace day_of_week_of_300th_day_of_previous_year_l508_508516

theorem day_of_week_of_300th_day_of_previous_year
  (current_year : bool)  -- True if current year is leap year, False otherwise
  (current_year_day_200 : ℕ) (next_year_day_100 : ℕ)
  (current_year_day_200_is_sunday : current_year_day_200 = 0)  -- Sunday is represented as 0
  (next_year_day_100_is_sunday : next_year_day_100 = 0) :
  (current_year = tt → (next_year_day_100 - current_year_day_200 + 365) % 7 = 0) →
  (current_year = ff → (next_year_day_100 - current_year_day_200 + 366) % 7 = 0) →
  (let previous_year_day_300 := (if current_year = tt then 300 - (365 % 7 + 200 % 7 + 7) else 300 - (366 % 7 + 200 % 7 + 7))
   in previous_year_day_300 % 7 = 1) :=
begin
  sorry
end

end day_of_week_of_300th_day_of_previous_year_l508_508516


namespace euler_convex_polyhedron_20_triangular_faces_l508_508943

/--
Given a convex polyhedron with 20 triangular faces,
prove that the number of vertices \( V \) is 12.
-/
theorem euler_convex_polyhedron_20_triangular_faces :
  ∃ V E, (∀ F : ℕ, F = 20) → (∀ f : ℕ, f = 3) →
  (∀ e : ℕ, e = 1.5 * 20) → (V - 30 + 20 = 2) ∧ V = 12 :=
by {
  sorry
}

end euler_convex_polyhedron_20_triangular_faces_l508_508943


namespace largest_number_among_given_largest_number_is_pi_l508_508030

theorem largest_number_among_given (x : ℝ) (hx : x ∈ {-5, -real.sqrt 3, 0, real.pi, 3}) : x ≤ real.pi :=
begin
  by { simp [hx], linarith [real.pi_gt_three, real.sqrt_pos_of_pos (by norm_num : (0 : ℝ) < 3)], },
end

theorem largest_number_is_pi : ∀ x ∈ {-5, -real.sqrt 3, 0, real.pi, 3}, x ≤ real.pi :=
λ x hx, largest_number_among_given x hx

end largest_number_among_given_largest_number_is_pi_l508_508030


namespace skeleton_6x6x6_cube_remove_to_skeleton_7x7x7_cube_l508_508670

-- Definitions based on conditions provided in the problem statement

/--
  The skeleton of a 6 × 6 × 6 cube consists of 56 unit cubes.
-/
theorem skeleton_6x6x6_cube : 
  (number_of_skeleton_cubes 6) = 56 := 
sorry

/--
  The number of unit cubes that must be removed from a 7 × 7 × 7 cube to obtain its skeleton is 275.
-/
theorem remove_to_skeleton_7x7x7_cube : 
  (number_of_cubes_to_remove_to_skeleton 7) = 275 := 
sorry

end skeleton_6x6x6_cube_remove_to_skeleton_7x7x7_cube_l508_508670


namespace difference_of_squares_65_35_l508_508763

theorem difference_of_squares_65_35 : 65^2 - 35^2 = 3000 := 
  sorry

end difference_of_squares_65_35_l508_508763


namespace noah_ava_zoo_trip_l508_508242

theorem noah_ava_zoo_trip :
  let tickets_cost := 5
  let bus_fare := 1.5
  let initial_money := 40
  let num_people := 2
  let round_trip := 2

  initial_money - (num_people * tickets_cost + num_people * round_trip * bus_fare) = 24 :=
by
  let tickets_cost := 5
  let bus_fare := 1.5
  let initial_money := 40
  let num_people := 2
  let round_trip := 2

  have cost := num_people * tickets_cost + num_people * round_trip * bus_fare
  have remaining := initial_money - cost
  have : remaining = 24, sorry
  exact this

end noah_ava_zoo_trip_l508_508242


namespace part1_inter_part1_complB_part2_l508_508496

def U := ℝ
def A := { x : ℝ | -4 ≤ x ∧ x < 2 }
def B := { x : ℝ | -1 < x ∧ x ≤ 3 }
def P := { x : ℝ | x ≤ 0 ∨ x ≥ 5 }

theorem part1_inter (x : ℝ) : x ∈ A ∩ B ↔ -1 < x ∧ x < 2 := by
  sorry

theorem part1_complB (x : ℝ) : x ∈ U \ B ↔ x > 3 ∨ x ≤ -1 := by
  sorry

theorem part2 (x : ℝ) : x ∈ (A ∩ B) ∪ (U \ P) ↔ 0 < x ∧ x < 2 := by
  sorry

end part1_inter_part1_complB_part2_l508_508496


namespace find_value_of_m_l508_508495

theorem find_value_of_m (x y m : ℝ) 
  (h1 : 2 * x + y = 7) 
  (h2 : x + 2 * y = m - 3) 
  (h3 : x - y = 2) : 
  m = 8 :=
begin
  sorry
end

end find_value_of_m_l508_508495


namespace length_AF_is_4_l508_508467

-- Define the focus and the parabola y^2 = 4x
def parabola (y x : ℝ) : Prop := y^2 = 4 * x

-- Define the focus of the parabola
def is_focus (F : ℝ × ℝ) : Prop := F = (1, 0)

-- Define a point on the parabola
def is_point_on_parabola (A : ℝ × ℝ) : Prop := ∃ y, parabola y (A.fst)

-- Define the midpoint condition
def midpoint_condition (A F : ℝ × ℝ) : Prop := (A.fst + F.fst) / 2 = 2

-- Define the distance function
noncomputable def distance (A F : ℝ × ℝ) : ℝ := 
  real.sqrt ((A.fst - F.fst)^2 + (A.snd - F.snd)^2)

-- The proof statement
theorem length_AF_is_4 (F : ℝ × ℝ) (A : ℝ × ℝ) 
  (h_focus : is_focus F)
  (h_point : is_point_on_parabola A)
  (h_midpoint : midpoint_condition A F) :
  distance A F = 4 :=
sorry

end length_AF_is_4_l508_508467


namespace parallelogram_equation_l508_508815

variable (a b c x y : ℝ)

theorem parallelogram_equation :
  (| b * x - (a + c) * y | + | b * x + (c - a) * y - b * c | = b * c) :=
sorry

end parallelogram_equation_l508_508815


namespace max_y_coordinate_polar_curve_l508_508792

theorem max_y_coordinate_polar_curve :
  ∀ (θ : ℝ), let r := cos (2 * θ),
                 x := r * cos θ,
                 y := r * sin θ in
  ∃ (θ_max : ℝ), y = (3 * real.sqrt 3) / 4 := 
by
  sorry -- Proof is omitted

end max_y_coordinate_polar_curve_l508_508792


namespace problem1_problem2_problem3_problem4_l508_508395

theorem problem1 : 3 * real.sqrt 8 - real.sqrt 32 = 2 * real.sqrt 2 := by
  sorry

theorem problem2 : (real.sqrt 6 * real.sqrt 2) / real.sqrt 3 = 2 := by
  sorry

theorem problem3 : (real.sqrt 24 + real.sqrt (1 / 6)) / real.sqrt 3 = (13 * real.sqrt 2) / 6 := by
  sorry

theorem problem4 : (real.sqrt 27 - real.sqrt 12) / real.sqrt 3 - 1 = 0 := by
  sorry

end problem1_problem2_problem3_problem4_l508_508395


namespace area_quadrilateral_ABCM_l508_508396

theorem area_quadrilateral_ABCM
  (A B C D E F G H I J : Point)
  (sides : ((A, B), (B, C), (C, D), (D, E), (E, F), (F, G), (G, H), (H, I), (I, J), (J, A)))
  (lengths : ∀ (p q : Point), (p, q) ∈ sides → ((p == A ∧ q == B) ∨ (p == C ∧ q == D) ∨ (p == E ∧ q == F) ∨ (p == G ∧ q == H) ∨ (p == I ∧ q == J)) → dist p q = 5 
                     ∧ ((p == B ∧ q == C) ∨ (p == D ∧ q == E) ∨ (p == F ∧ q == G) ∨ (p == H ∧ q == I) ∨ (p == J ∧ q == A)) → dist p q = 3)
  (right_angles : ∀ (p q r : Point), (p, q) ∈ sides ∧ (q, r) ∈ sides → angle p q r = π / 2)
  (M : Point)
  (intersect_M : lies_on M (line_through A E) ∧ lies_on M (line_through C G)) :
  area_quadrilateral A B C M = 40 :=
sorry

end area_quadrilateral_ABCM_l508_508396


namespace day_of_week_of_300th_day_of_previous_year_l508_508515

theorem day_of_week_of_300th_day_of_previous_year
  (current_year : bool)  -- True if current year is leap year, False otherwise
  (current_year_day_200 : ℕ) (next_year_day_100 : ℕ)
  (current_year_day_200_is_sunday : current_year_day_200 = 0)  -- Sunday is represented as 0
  (next_year_day_100_is_sunday : next_year_day_100 = 0) :
  (current_year = tt → (next_year_day_100 - current_year_day_200 + 365) % 7 = 0) →
  (current_year = ff → (next_year_day_100 - current_year_day_200 + 366) % 7 = 0) →
  (let previous_year_day_300 := (if current_year = tt then 300 - (365 % 7 + 200 % 7 + 7) else 300 - (366 % 7 + 200 % 7 + 7))
   in previous_year_day_300 % 7 = 1) :=
begin
  sorry
end

end day_of_week_of_300th_day_of_previous_year_l508_508515


namespace find_m_l508_508862

variables (m : ℝ)

-- define the vectors
def a := (1 : ℝ, m)
def b := (2 : ℝ,  5)
def c := (m, 3)

-- define vector addition and subtraction
def vec_add (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)
def vec_sub (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 - v.1, u.2 - v.2)
def parallel (u v : ℝ × ℝ) : Prop := (u.1 * v.2 = u.2 * v.1)

-- given conditions
def conditions : Prop := parallel (vec_add a c) (vec_sub a b)

-- the final goal
theorem find_m (h : conditions m) : m = (3 + Real.sqrt 17) / 2 ∨ m = (3 - Real.sqrt 17) / 2 := 
sorry

end find_m_l508_508862


namespace problem_statement_l508_508091

noncomputable theory

-- Define the polynomials g(x) and f(x) with given conditions
def g (p : ℝ) : Polynomial ℝ := X^3 + (C p) * X^2 + 2 * X + 20
def f (p q r : ℝ) : Polynomial ℝ := X^4 + 2 * X^3 + (C q) * X^2 + 200 * X + (C r)

-- Define the condition that g(p) has three distinct roots
def has_three_distinct_roots {R : Type*} [CommRing R] (P : Polynomial R) : Prop :=
  ∃ a b c : R, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ P.factor_multiset = Multiset.of_list [a, b, c]

-- Define the main statement that needs to be proved
theorem problem_statement (p : ℝ) (q : ℝ) (r : ℝ) (h_roots : has_three_distinct_roots (g p))
  (h_factor : ∀ x, g p = 0 → f p q r x = 0) : f p q r (-1) = -6319 := 
sorry

end problem_statement_l508_508091


namespace scientific_notation_correct_l508_508922

theorem scientific_notation_correct (n : ℕ) (h : n = 11580000) : n = 1.158 * 10^7 := 
sorry

end scientific_notation_correct_l508_508922


namespace max_modulus_z3_add_3z_add_2i_l508_508812

open Complex

theorem max_modulus_z3_add_3z_add_2i (z : ℂ) (hz : abs z = 1) :
  ∃ θ : ℝ, z = exp (θ * I) ∧ ∀ z : ℂ, abs z = 1 → abs (z^3 + 3*z + 2*I) ≤ 4*Real.sqrt 2 := 
begin
  use ∃ θ : ℝ, z = exp (θ * I),
  sorry
end

end max_modulus_z3_add_3z_add_2i_l508_508812


namespace bus_speed_excluding_stoppages_l508_508414

theorem bus_speed_excluding_stoppages (v : ℝ) (stoppage_time : ℝ) (speed_incl_stoppages : ℝ) :
  stoppage_time = 15 / 60 ∧ speed_incl_stoppages = 48 → v = 64 :=
by
  intro h
  sorry

end bus_speed_excluding_stoppages_l508_508414


namespace jerry_games_l508_508954

theorem jerry_games (initial_games : ℕ) (percentage_increase : ℝ) (new_games : ℕ) : 
  initial_games = 7 → 
  percentage_increase = 0.30 → 
  new_games = floor (percentage_increase * initial_games).to_real → 
  initial_games + new_games = 9 := 
by 
  intros h1 h2 h3
  sorry

end jerry_games_l508_508954


namespace delores_initial_money_l508_508772

theorem delores_initial_money (cost_computer : ℕ) (cost_printer : ℕ) (money_left : ℕ) (initial_money : ℕ) :
  cost_computer = 400 → cost_printer = 40 → money_left = 10 → initial_money = cost_computer + cost_printer + money_left → initial_money = 450 := by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end delores_initial_money_l508_508772


namespace find_moles_of_NaCl_l508_508084

-- Define the chemical reaction as an equation
def chemical_reaction (NaCl KNO3 NaNO3 KCl : ℕ) : Prop :=
  NaCl + KNO3 = NaNO3 + KCl

-- Define the problem conditions
def problem_conditions (naCl : ℕ) : Prop :=
  ∃ (kno3 naNo3 kcl : ℕ),
    kno3 = 3 ∧
    naNo3 = 3 ∧
    chemical_reaction naCl kno3 naNo3 kcl

-- Define the goal statement
theorem find_moles_of_NaCl (naCl : ℕ) : problem_conditions naCl → naCl = 3 :=
by
  sorry -- proof to be filled in later

end find_moles_of_NaCl_l508_508084


namespace custom_op_identity_l508_508769

def custom_op (x y : ℕ) : ℕ := x * y + 3 * x - 4 * y

theorem custom_op_identity : custom_op 7 5 - custom_op 5 7 = 14 :=
by
  sorry

end custom_op_identity_l508_508769


namespace paul_books_l508_508996

theorem paul_books
  (initial_books : ℕ)
  (prices : ℕ → ℕ)
  (total_spent : ℕ)
  (number_of_categories : ℕ)
  (mystery_deal : ℕ → ℕ)
  (sci_fi_discount : ℚ)
  (romance_price : ℕ)
  (final_books : ℕ) :
  initial_books = 50 ∧
  prices 0 = 4 ∧
  prices 1 = 6 ∧
  prices 2 = 5 ∧
  total_spent = 90 ∧
  number_of_categories = 3 ∧
  (∀ x, mystery_deal x = 2 / 3 * x) ∧
  (x = 20 / 100 * 6 : ℚ) ∧
  (romance_price = 5) →
  final_books = 21 :=
sorry

end paul_books_l508_508996


namespace remainder_when_even_coefficients_divided_by_3_eq_1_l508_508896

theorem remainder_when_even_coefficients_divided_by_3_eq_1
  (n : ℕ) (h_pos : 0 < n)
  (a : ℕ → ℤ) 
  (h : (2*x + 4)^(2*n) = ∑ i in range (2*n + 1), a i * x^i) :
  ((∑ k in range (n + 1), a (2 * k)) % 3) = 1 := by
  sorry

end remainder_when_even_coefficients_divided_by_3_eq_1_l508_508896


namespace angle_between_vectors_l508_508140

variable {a b : ℝ × ℝ}

def is_nonzero (v : ℝ × ℝ) : Prop := v ≠ (0, 0)

def magnitude (v : ℝ × ℝ) : ℝ := real.sqrt (v.1 ^ 2 + v.2 ^ 2)

def is_perpendicular (v1 v2 : ℝ × ℝ) : Prop := v1.1 * v2.1 + v1.2 * v2.2 = 0

theorem angle_between_vectors :
  is_nonzero a →
  is_nonzero b →
  magnitude a = 2 →
  magnitude b = 4 →
  is_perpendicular (a.1 + b.1, a.2 + b.2) a →
  real.arccos ((a.1 * b.1 + a.2 * b.2) / (magnitude a * magnitude b)) = real.pi * 2 / 3 :=
begin
  intros,
  sorry
end

end angle_between_vectors_l508_508140


namespace ellipse_focus_distance_l508_508821

theorem ellipse_focus_distance :
  ∀ (P : ℝ × ℝ), (P.1^2 / 25 + P.2^2 / 9 = 1) →
    (P.1 / (5/4)^2 = 5) →
    sqrt (25 - (25 * (1 - (9 / 25)))) = 4 →
    sqrt (25 - (25 * (1 - (9 / 25)))) + 6 = 10 → 
    True :=
sorry

end ellipse_focus_distance_l508_508821


namespace non_empty_intersection_l508_508563

variable {X : Type*} -- Let X be a type.

-- Define the family of r-element subsets of X.
variables {t r : ℕ}
variable (A : Fin t → Finset X)
variable (h₁ : ∀ i : Fin t, (A i).card = r)

-- Condition: The intersection of every r+1 subsets in \mathscr{A} is non-empty.
variable (h₂ : ∀ (s : Finset (Fin t)), s.card = r + 1 → s.bUnion A ≠ ∅)

-- Conclusion: The intersection of all subsets in \mathscr{A} is non-empty.
theorem non_empty_intersection (A : Fin t → Finset X) (h₁ : ∀ i : Fin t, (A i).card = r)
    (h₂ : ∀ s : Finset (Fin t), s.card = r + 1 → s.bUnion A ≠ ∅) : (Finset.inter (Finset.univ.map A) ≠ ∅) :=
by
  sorry

end non_empty_intersection_l508_508563


namespace floor_ceil_product_l508_508407

theorem floor_ceil_product :
  let f : ℕ → ℤ := λ n, (Int.floor (- (n : ℤ) - 0.5)) * (Int.ceil ((n : ℤ) + 0.5))
  let product : ℤ := ∏ i in Finset.range 7, -(i + 1)^2
  ∑ n in Finset.range 7, f n = product :=
by
  sorry

end floor_ceil_product_l508_508407


namespace rounding_to_two_decimal_places_l508_508737

theorem rounding_to_two_decimal_places (x : ℝ) (h : x = 2.7982) : 
  Real.approx x 0.01 = 2.80 :=
by 
  rw h
  exact Real.approximation_rounding_method 2.7982 0.01
  sorry

end rounding_to_two_decimal_places_l508_508737


namespace problem2024_l508_508054

theorem problem2024 :
  (∑ k in Finset.range 2023 | (2024 - k) / (k + 1)) / (∑ k in (Finset.range 2023) + 1 / (k + 2)) = 2024 := sorry

end problem2024_l508_508054


namespace sub_decimal_proof_l508_508785

theorem sub_decimal_proof : 2.5 - 0.32 = 2.18 :=
  by sorry

end sub_decimal_proof_l508_508785


namespace simplify_root_expression_l508_508678

theorem simplify_root_expression : (sqrt 27 + sqrt 243) / sqrt 48 = 3 := by
  sorry

end simplify_root_expression_l508_508678


namespace y_intercepts_of_curve_l508_508790

theorem y_intercepts_of_curve :
  (∃ (y : ℝ), 3 * 0 + 5 * y^2 = 25) ∧ (∀ (x y : ℝ), x = 0 → 3 * x + 5 * y^2 = 25 → (y = sqrt 5 ∨ y = -sqrt 5)) := 
by 
  sorry

end y_intercepts_of_curve_l508_508790


namespace evaluate_expression_l508_508782

theorem evaluate_expression :
  (3 ^ 1002 + 7 ^ 1003) ^ 2 - (3 ^ 1002 - 7 ^ 1003) ^ 2 = 56 * 10 ^ 1003 :=
by
  sorry

end evaluate_expression_l508_508782


namespace find_a_l508_508125

-- Define the function f(x)
def f (x : ℕ) (a : ℤ) := Real.log (x ^ 2 + a) / Real.log 2

-- State the problem with the necessary conditions and conclusion
theorem find_a (h : f 3 a = 1) : a = -7 :=
sorry

end find_a_l508_508125


namespace distinct_valid_c_values_l508_508802

theorem distinct_valid_c_values : 
  let is_solution (c : ℤ) (x : ℚ) := (5 * ⌊x⌋₊ + 3 * ⌈x⌉₊ = c) 
  ∃ s : Finset ℤ, (∀ c ∈ s, (∃ x : ℚ, is_solution c x)) ∧ s.card = 500 :=
by sorry

end distinct_valid_c_values_l508_508802


namespace smallest_x_value_l508_508560

-- Define the conditions
def is_isosceles_trapezoid (A B C D : ℝ) (AB CD AD BC : ℝ) : Prop :=
  AB > CD ∧ AD = BC ∧ (AD = (CD + AB) / 2)

def circle_center_on_AB_tangent_to_AD_BC (radius x : ℝ) : Prop :=
  ∃ (M : ℝ), (M = AB / 2) ∧ (radius = sqrt ((x ^ 2 - M ^ 2) / 2))

-- The proof statement
theorem smallest_x_value (AB CD AD BC : ℝ) (x radius : ℝ) (ht : is_isosceles_trapezoid A B C D AB CD AD BC)
    (hc : circle_center_on_AB_tangent_to_AD_BC radius x) :
    x = sqrt 2812.5 :=
by
  sorry

end smallest_x_value_l508_508560


namespace triangle_pqr_xy_l508_508306

noncomputable def semi_perimeter (a b c : ℝ) : ℝ := (a + b + c) / 2
noncomputable def area (a b c s : ℝ) : ℝ := Real.sqrt (s * (s - a) * (s - b) * (s - c))
noncomputable def inradius (A s : ℝ) : ℝ := A / s
noncomputable def height_from_p (A a : ℝ) : ℝ := (2 * A) / a
noncomputable def similarity_ratio (h₁ h₂ : ℝ) : ℝ := h₁ / h₂
noncomputable def length_xy (ratio qr : ℝ) : ℝ := ratio * qr
noncomputable def rel_prime_sum (m n : ℕ) : ℕ := m + n
noncomputable def gcd (a b : ℕ) : ℕ :=
  if b = 0 then a else gcd b (a % b)

theorem triangle_pqr_xy
  (PQ PR QR : ℝ)
  (PQ_eq : PQ = 13)
  (PR_eq : PR = 14)
  (QR_eq : QR = 15) :
  ∃ m n : ℕ, (gcd m n = 1) ∧ rel_prime_sum m n = 149 :=
by
  sorry

end triangle_pqr_xy_l508_508306


namespace cans_purchased_l508_508739

theorem cans_purchased (S Q E : ℝ) (h1 : Q ≠ 0) (h2 : S > 0) :
  (10 * E * S) / Q = (10 * (E : ℝ) * (S : ℝ)) / (Q : ℝ) := by 
  sorry

end cans_purchased_l508_508739


namespace find_smallest_n_l508_508182

noncomputable def P (n : ℕ) : ℝ :=
  (∏ k in finset.range (n - 1) + 1, (2 : ℝ) * k / (2 * k + 1)) * (1 / (2 * n + 1))

theorem find_smallest_n (h: 1 ≤ 1000) :
  ∃ n : ℕ, P n < 1 / 1000 ∧ (∀ m : ℕ, 1 ≤ m < n → P m ≥ 1 / 1000) :=
by
  sorry

end find_smallest_n_l508_508182


namespace max_value_ineq_l508_508843

theorem max_value_ineq (x y : ℝ) (hx1 : -5 ≤ x) (hx2 : x ≤ -3) (hy1 : 1 ≤ y) (hy2 : y ≤ 3) : 
  (x + y) / (x - 1) ≤ 2 / 3 := 
sorry

end max_value_ineq_l508_508843


namespace smallest_k_square_divisible_l508_508730

theorem smallest_k_square_divisible (k : ℤ) (n : ℤ) (h1 : k = 60)
    (h2 : ∀ m : ℤ, m < k → ∃ d : ℤ, d ∣ (k^2) → m = d ) : n = 3600 :=
sorry

end smallest_k_square_divisible_l508_508730


namespace fractional_part_sum_l508_508575

noncomputable def fractional_part (x : ℚ) : ℚ := x - ⌊x⌋

theorem fractional_part_sum (n a b : ℤ) (hn : n > 0) (ha_coprime : Nat.coprime a.natAbs n.natAbs) :
  (∑ k in Finset.range n, fractional_part ((↑k * a + b) / n)) = (n - 1) / 2 :=
by
    sorry

end fractional_part_sum_l508_508575


namespace min_value_of_f_l508_508462

noncomputable def f (x y z : ℝ) : ℝ :=
  x^2 / (1 + x) + y^2 / (1 + y) + z^2 / (1 + z)

theorem min_value_of_f (a b c x y z : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : c * y + b * z = a) (h2 : a * z + c * x = b) (h3 : b * x + a * y = c) :
  ∃ x y z : ℝ, f x y z = 1 / 2 := sorry

end min_value_of_f_l508_508462


namespace area_of_D_volume_of_rotated_D_l508_508092

-- Define the problem conditions
variables (a : ℝ) (ha : 0 < a)

-- Define the curve and line that determine region D
def curve (x y : ℝ) := (Real.sqrt x + Real.sqrt y = Real.sqrt a)
def line (x y : ℝ) := (x + y = a)

-- Define the region D and calculate the area and volume
def region_D := {p : ℝ × ℝ | curve p.1 p.2 ∨ line p.1 p.2}

-- Part 1: Area of D
theorem area_of_D : ∫ x in 0..(a/Real.sqrt 2), (a/Real.sqrt 2 - (x^2/(a*Real.sqrt 2) + a/(2*Real.sqrt 2))) = a^2 / 3 :=
sorry

-- Part 2: Volume of the solid obtained by rotating D about the line x + y = a
theorem volume_of_rotated_D : 
  let volume := 2 * ∫ x in 0..(a/Real.sqrt 2), π * (a/(2*Real.sqrt 2) - x^2/(a*Real.sqrt 2))^2 
  in volume = (π * Real.sqrt 2 / 15) * a^3 :=
sorry

end area_of_D_volume_of_rotated_D_l508_508092


namespace find_circle_equation_find_line_equation_l508_508844

-- First Problem: Finding the standard equation of circle M
theorem find_circle_equation 
  (t : ℝ) (x y : ℝ) 
  (h1 : y = 1/2 * x) 
  (h2 : x*y = t) 
  (h3 : 2 * sqrt((2 * t)^2 - t^2) = 2 * sqrt(3)) :
  (x - 2)^2 + (y - 1)^2 = 4 := 
by
  intros
  sorry

-- Second Problem: Finding the equation of line l
theorem find_line_equation 
  (k : ℝ) (x y A B : ℝ) 
  (h1 : y = 1/2 * x) 
  (h2 : x*y = t) 
  (h3 : 2 * sqrt((2 * t)^2 - t^2) = 2 * sqrt(3)) 
  (h4 : x = 2 ∧ y = 2) 
  (h5 : (2 * k) - y - 2 * k + 2 = 0) 
  (h6 : x = 2 ∧ y = -1) 
  (h7 : (1/2) * (abs(3)/(sqrt(k^2 + 1)) * abs(2 * sqrt((4*k^2 + 3)/(k^2 + 1))) == 3 * sqrt(3)) :
  y = 2 :=
by
  intros
  sorry

end find_circle_equation_find_line_equation_l508_508844


namespace athena_fruit_drinks_l508_508035

theorem athena_fruit_drinks :
  ∃ x : ℝ, 3 * 3 + 2.5 * x = 14 ∧ x = 2 :=
by
  use 2
  split
  { -- Prove the equation
    have h1 : 3 * 3 = 9 := by norm_num
    have h2 : 2.5 * 2 = 5 := by norm_num
    rw [h1, h2]
    norm_num
  }
  { -- Prove x = 2
    norm_num
  }
  sorry

end athena_fruit_drinks_l508_508035


namespace decagon_product_l508_508708

noncomputable def point := ℂ

def Q1 : point := (2 : ℝ) + 0 * I
def Q6 : point := (4 : ℝ) + 0 * I
def center : point := (3 : ℝ) + 0 * I

-- The vertices of the regular decagon are roots of (z - center)^10 = 1
def vertices : Fin 10 → point := fun k => center + exp (2 * Real.pi * I * k / 10)

theorem decagon_product :
  (∏ k, vertices k) = 59048 := by
  sorry

end decagon_product_l508_508708


namespace inequality_proof_l508_508556

theorem inequality_proof (n : ℕ) (h_n : n ≥ 3) 
  (x : ℕ → ℝ) (h_order : ∀ i, 1 ≤ i → i < n → x i < x (i + 1)) :
  (n * (n - 1) / 2) * 
  (Finset.sum (Finset.Ico 1 n) (λ i, Finset.sum (Finset.Ico (i + 1) (n + 1)) (λ j, x i * x j))) >
  (Finset.sum (Finset.range (n - 1)) (λ i, (n - 1 - i) * x (i + 1))) *
  (Finset.sum (Finset.range n) (λ j, if j = 0 then 0 else (j * x j))):=
sorry

end inequality_proof_l508_508556


namespace clownfish_in_display_tank_l508_508383

theorem clownfish_in_display_tank (C B : ℕ) (h1 : C = B) (h2 : C + B = 100) : 
  (B - 26 - (B - 26) / 3) = 16 := by
  sorry

end clownfish_in_display_tank_l508_508383


namespace changing_quantities_l508_508603

def midpoint (A B : Point) : Point := { x := (A.x + B.x) / 2, y := (A.y + B.y) / 2 }

theorem changing_quantities (P A B : Point) (θ : ℝ) :
  let M := midpoint P A;
  let N := midpoint P B;
  let dPM := P = M + δ * (cos θ, sin θ);
  let dPN := P = N + δ * (cos θ, sin θ);
  P.x = P₀.x + δ * cos θ ∧ P.y = P₀.y + δ * sin θ → 
  (∃ M N PM PN AB MN perimeter ∆PAB A_trapezoid : ℝ, 
    ¬MN_varies ∧
    perimeter_varies ∧
    area_varies ∧
    trap_area_varies)
  sorry

end changing_quantities_l508_508603


namespace cookies_flour_and_eggs_l508_508707

theorem cookies_flour_and_eggs (c₁ c₂ : ℕ) (f₁ f₂ : ℕ) (e₁ e₂ : ℕ) 
  (h₁ : c₁ = 40) (h₂ : f₁ = 3) (h₃ : e₁ = 2) (h₄ : c₂ = 120) :
  f₂ = f₁ * (c₂ / c₁) ∧ e₂ = e₁ * (c₂ / c₁) :=
by
  sorry

end cookies_flour_and_eggs_l508_508707


namespace tangent_points_l508_508001

noncomputable def f (x : ℝ) : ℝ := x^3 + 1
def P : ℝ × ℝ := (-2, 1)
noncomputable def f' (x : ℝ) : ℝ := 3 * x^2

theorem tangent_points (x0 : ℝ) (y0 : ℝ) (hP : P = (-2, 1)) (hf : y0 = f x0) :
  (3 * x0^2 = (y0 - 1) / (x0 + 2)) → (x0 = 0 ∨ x0 = -3) :=
by
  sorry

end tangent_points_l508_508001


namespace selling_price_decreased_l508_508719

theorem selling_price_decreased (d m : ℝ) (hd : d = 0.10) (hm : m = 0.10) :
  (1 - d) * (1 + m) < 1 :=
by
  rw [hd, hm]
  sorry

end selling_price_decreased_l508_508719


namespace length_PQ_eq_zero_l508_508561

theorem length_PQ_eq_zero :
  ∀ (E F G H E' F' G' H' P Q : ℝ × ℝ × ℝ),
    E = (0, 0, 0) → F = (2, 0, 0) → G = (2, 1, 0) → H = (0, 1, 0) →
    E' = (0, 0, 12) → F' = (2, 0, 6) → G' = (2, 1, 15) → H' = (0, 1, 21) →
    P = ((E'.1 + G'.1) / 2, (E'.2 + G'.2) / 2, (E'.3 + G'.3) / 2) →
    Q = ((F'.1 + H'.1) / 2, (F'.2 + H'.2) / 2, (F'.3 + H'.3) / 2) →
    dist P Q = 0 :=
by
  intro E F G H E' F' G' H' P Q hE hF hG hH hE' hF' hG' hH' hP hQ
  sorry

end length_PQ_eq_zero_l508_508561


namespace calc_area_quad_l508_508946

noncomputable def area_of_quad (A B C D : Type) [metric_space A] 
  (angle_BCD : real_arccos (-1 / 2)) -- 120 degrees cosine equivalent
  (AB : ℝ) (BC : ℝ) (CD : ℝ) (AD : ℝ) : ℝ :=
  let BD := real.sqrt (BC^2 + CD^2 + BC * CD) in 
  let area_BCD := (real.sqrt 3 / 4 * BC * CD) in
  let s := (AB + BD + AD) / 2 in
  let area_BDA := real.sqrt (s * (s - AB) * (s - BD) * (s - AD)) in
  area_BCD + area_BDA

theorem calc_area_quad  (A B C D : Type) [metric_space A]
  (angle_BCD : real_arccos (-1 / 2)) -- 120 degrees cosine equivalent
  (AB : 13) (BC : 6) (CD : 5) (AD : 12) :
  area_of_quad A B C D angle_BCD AB BC CD AD = real.sqrt 3 / 2 * 15 + area_of_quad A B D BAD :=
sorry

end calc_area_quad_l508_508946


namespace system1_solution_system2_solution_l508_508258

-- Definition and proof for System (1)
theorem system1_solution (x y : ℝ) (h1 : x - y = 2) (h2 : 2 * x + y = 7) : x = 3 ∧ y = 1 := 
by 
  sorry

-- Definition and proof for System (2)
theorem system2_solution (x y : ℝ) (h1 : x - 2 * y = 3) (h2 : (1 / 2) * x + (3 / 4) * y = 13 / 4) : x = 5 ∧ y = 1 :=
by 
  sorry

end system1_solution_system2_solution_l508_508258


namespace max_y_coordinate_on_graph_of_cos_2theta_l508_508794

theorem max_y_coordinate_on_graph_of_cos_2theta :
  (∃ θ, (θ ∈ Set.Icc (-Real.pi) Real.pi) ∧ 
        ∀ θ', θ' ∈ Set.Icc (-Real.pi) Real.pi → 
        let y := Real.cos (2 * θ') * Real.sin θ' in 
          y ≤ Real.cos (2 * θ) * Real.sin θ) → 
  ∃ θ, θ ∈ Set.Icc (-Real.pi) Real.pi ∧ Real.cos (2 * θ) * Real.sin θ = (2 * Real.sqrt 6) / 3 := 
sorry

end max_y_coordinate_on_graph_of_cos_2theta_l508_508794


namespace tic_tac_toe_winning_boards_l508_508742

-- Define the board as a 4x4 grid
def Board := Array (Array (Option Bool))

-- Define a function that returns all possible board states after 3 moves
noncomputable def numberOfWinningBoards : Nat := 140

theorem tic_tac_toe_winning_boards:
  numberOfWinningBoards = 140 :=
by
  sorry

end tic_tac_toe_winning_boards_l508_508742


namespace black_haired_girls_count_l508_508988

-- Definitions from the conditions in the problem
def total_initial_girls := 120
def initial_blonde_girls := 50
def added_blonde_girls := 25

-- The statement to prove
theorem black_haired_girls_count : 
  let initial_black_girls := total_initial_girls - initial_blonde_girls in
  initial_black_girls = 70 := by
  sorry

end black_haired_girls_count_l508_508988


namespace inverse_of_B_cubed_l508_508107

theorem inverse_of_B_cubed (B_inv : Matrix (Fin 2) (Fin 2) ℤ) 
  (hB_inv : B_inv = (Matrix.of 2 2 ![[3, -1], [1, 1]])) :
  (B_inv ^ 3).inv = (Matrix.of 2 2 ![[20, -12], [12, -4]]) :=
by
  sorry

end inverse_of_B_cubed_l508_508107


namespace num_right_angled_triangles_l508_508472

theorem num_right_angled_triangles {A B C D E F H : Type} 
  [triangle : triangle A B C]
  (h1 : acute_triangle A B C)
  (h2 : altitude A D H)
  (h3 : altitude B E H)
  (h4 : altitude C F H)
  (h5 : orthocenter A B C H):
  num_right_angled_triangles A B C D E F H = 12 := sorry

end num_right_angled_triangles_l508_508472


namespace find_y_l508_508005

noncomputable def x : ℝ := 0.7142857142857143

def equation (y : ℝ) : Prop :=
  (x * y) / 7 = x^2

theorem find_y : ∃ y : ℝ, equation y ∧ y = 5 :=
by
  use 5
  have h1 : x != 0 := by sorry
  have h2 : (x * 5) / 7 = x^2 := by sorry
  exact ⟨h2, rfl⟩

end find_y_l508_508005


namespace smallest_k_divides_polynomial_l508_508088

theorem smallest_k_divides_polynomial :
  ∃ k : ℕ, 0 < k ∧ (∀ z : ℂ, (z^10 + z^9 + z^8 + z^6 + z^5 + z^4 + z + 1) ∣ (z^k - 1)) ∧ k = 84 :=
by
  sorry

end smallest_k_divides_polynomial_l508_508088


namespace triangle_length_RS_l508_508069

noncomputable def length_RS (P Q R S : Point) (side_length : ℝ) : Prop :=
  let PR : ℝ := side_length
  let PQ : ℝ := side_length
  let QR : ℝ := side_length
  let PQR_perimeter : ℝ := 3 * side_length
  let PRS_perimeter : ℝ := PR + distance R S + distance S P
  (isEquilateral P Q R side_length) ∧
  (isRightTriangle P R S) ∧
  (PQR_perimeter = PRS_perimeter) ∧
  (distance R S = 3)

-- Define Points according to the conditions.
variables {P Q R S : Point} {side_length : ℝ}

-- Define the key conditions of the problem.
axiom isEquilateral (P Q R : Point) (side_length : ℝ) : Prop
axiom isRightTriangle (P R S : Point) : Prop
axiom distance : Point → Point → ℝ

-- Define the problem statement as a theorem to prove.
theorem triangle_length_RS (h1 : isEquilateral P Q R side_length)
                          (h2 : isRightTriangle P R S)
                          (h3 : 3 * side_length = side_length + distance R S + distance S P) :
  distance R S = 3 :=
sorry

end triangle_length_RS_l508_508069


namespace complement_of_M_l508_508211

-- Definitions:
def U : Set ℝ := Set.univ
def M : Set ℝ := {x | -1 < x ∧ x ≤ 2}

-- Assertion:
theorem complement_of_M :
  (U \ M) = {x | x ≤ -1} ∪ {x | 2 < x} :=
by sorry

end complement_of_M_l508_508211


namespace ball_distribution_l508_508153

theorem ball_distribution : 
  ∃ (ways : ℕ), ways = 2 ∧ 
  ( ∀ (balls boxes : ℕ), balls = 6 ∧ boxes = 4 → 
  ∃ (distribution : list ℕ), distribution.length = boxes ∧ 
  distribution.sum = balls ∧ 
  ∀ (n : ℕ), n ∈ distribution → n ≥ 1  ) :=
sorry

end ball_distribution_l508_508153


namespace right_angle_at_M_l508_508247

-- Define the triangle ABC and points K, L on AB
variables {A B C K L M : Type} [metric_space M] [metric_space K] 
  (triangle : is_triangle A B C) (hK : ∈ A B K) (hL : ∈ A B L)

-- Define the conditions AK = LB and KL = BC
variables (h1 : dist A K = dist B L) (h2 : dist K L = dist B C)

-- Define point M as the midpoint of AC
variables (M : is_midpoint A C)

-- Main theorem statement
theorem right_angle_at_M : ∠ K M L = 90° :=
sorry

end right_angle_at_M_l508_508247


namespace range_of_a_l508_508521

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, f x = ax^2 + log x ∧ x > 0 ∧ (2 * a * x + 1/x) → ∞)
  ↔ a < 0 :=
sorry

end range_of_a_l508_508521


namespace intersection_A_B_l508_508164

def A : Set ℝ := {x | (2 * x + 1) * (x - 3) < 0}

def B : Set ℕ := {x | x ≤ 5}

def N_star : Set ℕ := {x | x > 0}

theorem intersection_A_B : A ∩ (B ∩ N_star) = {1, 2} :=
by
  sorry

end intersection_A_B_l508_508164


namespace count_three_digit_integers_with_product_thirty_l508_508501

theorem count_three_digit_integers_with_product_thirty :
  (∃ S : Finset (ℕ × ℕ × ℕ),
      (∀ (a b c : ℕ), (a, b, c) ∈ S → a * b * c = 30 ∧ 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 1 ≤ c ∧ c ≤ 9) 
    ∧ S.card = 12) :=
by
  sorry

end count_three_digit_integers_with_product_thirty_l508_508501


namespace chord_length_of_concentric_circles_l508_508619

noncomputable def radius_larger : ℝ := sorry
noncomputable def radius_smaller : ℝ := sorry
noncomputable def chord_length : ℝ := 5 * Real.sqrt 2

lemma annulus_area (R r : ℝ) (h : R^2 - r^2 = 25 / 2) : 12.5 * Real.pi = 𝜋 * (R^2 - r^2) :=
by sorry

lemma chord_length_tangent (R r : ℝ) (h : R^2 - r^2 = 25 / 2) : 
  let c := 2 * (Real.sqrt (R^2 - r^2))
  in c = 5 * Real.sqrt 2 :=
by sorry

theorem chord_length_of_concentric_circles : chord_length = 5 * Real.sqrt 2 :=
by {
  let R := radius_larger,
  let r := radius_smaller,
  have h : R^2 - r^2 = 25 / 2 := by sorry,
  exact chord_length_tangent R r h,
}

end chord_length_of_concentric_circles_l508_508619


namespace blue_eyed_kitten_percentage_is_correct_l508_508593

def total_blue_eyed_kittens : ℕ := 5 + 6 + 4 + 7 + 3

def total_kittens : ℕ := 12 + 16 + 11 + 19 + 12

def percentage_blue_eyed_kittens (blue : ℕ) (total : ℕ) : ℚ := (blue : ℚ) / (total : ℚ) * 100

theorem blue_eyed_kitten_percentage_is_correct :
  percentage_blue_eyed_kittens total_blue_eyed_kittens total_kittens = 35.71 := sorry

end blue_eyed_kitten_percentage_is_correct_l508_508593


namespace pigs_count_l508_508930

-- Definitions from step a)
def pigs_leg_count : ℕ := 4 -- Each pig has 4 legs
def hens_leg_count : ℕ := 2 -- Each hen has 2 legs

variable {P H : ℕ} -- P is the number of pigs, H is the number of hens

-- Condition from step a) as a function
def total_legs (P H : ℕ) : ℕ := pigs_leg_count * P + hens_leg_count * H
def total_heads (P H : ℕ) : ℕ := P + H

-- Theorem to prove the number of pigs given the condition
theorem pigs_count {P H : ℕ} (h : total_legs P H = 2 * total_heads P H + 22) : P = 11 :=
  by 
    sorry

end pigs_count_l508_508930


namespace bc_sum_l508_508717

theorem bc_sum (A B C : ℝ) (h1 : A + B + C = 500) (h2 : A + C = 200) (h3 : C = 10) : B + C = 310 := by
  sorry

end bc_sum_l508_508717


namespace number_of_true_statements_l508_508722

theorem number_of_true_statements :
  let stmt1 := false -- incorrect statement about sampling method
  let stmt2 := true  -- correct negation of existential
  let stmt3 := false -- incorrect condition about logarithm function
  let stmt4 := true  -- correct converse statement about opposite numbers
  (∑ b in [stmt1, stmt2, stmt3, stmt4], if b then 1 else 0) = 2 := 
by 
  sorry

end number_of_true_statements_l508_508722


namespace main_theorem_l508_508581

variables {U : Type} [fintype U] (f g : U → U)
variables (S T : set U)
variables [bijective f] [bijective g]

def S : set U := {w : U | f (f w) = g (g w)}
def T : set U := {w : U | f (g w) = g (f w)}

theorem main_theorem 
        (hU : ∀ w : U, w ∈ S ∨ w ∈ T) : 
        ∀ w: U, (f w ∈ S) ↔ (g w ∈ S) :=
sorry

end main_theorem_l508_508581


namespace competition_sequences_l508_508263

-- Define the problem conditions
def team_size : Nat := 7

-- Define the statement to prove
theorem competition_sequences :
  (Nat.choose (2 * team_size) team_size) = 3432 :=
by
  -- Proof will go here
  sorry

end competition_sequences_l508_508263


namespace range_of_x0_l508_508850

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 3^(x + 1) else log 2 x

theorem range_of_x0 (x0 : ℝ) (h : f x0 ≥ 1) : 
  x0 ∈ set.Icc -1 0 ∪ set.Ici 2 :=
by {
  intros,
  sorry
}

end range_of_x0_l508_508850


namespace problem_statement_l508_508056

-- Condition definitions
def numerator : ℚ := ∑ k in Finset.range 2023, (2024 - (k + 1)) / (k + 1)
def denominator : ℚ := ∑ k in Finset.range (2024 - 2 + 1), 1 / (k + 2)

-- The statement to prove
theorem problem_statement : 
  (numerator / denominator) = 2024 := 
by sorry

end problem_statement_l508_508056


namespace no_two_adjacent_odd_vertices_exist_l508_508675

-- Let V, E, and F be the number of vertices, edges, and faces respectively in the triangulation
def sphere_triangulation (V E F : ℕ) : Prop :=
  V - E + F = 2 ∧ ∀ v, v ∈ vertices → (degree v = 3)

-- Defining the problem conditions
def three_neighbor_countries (countries : Type) (borders : countries → set countries) (triangulated_sphere : Prop) : Prop :=
  ∀ c, c ∈ countries → #|borders c| = 3

-- The main problem that needs to be proved
theorem no_two_adjacent_odd_vertices_exist (countries : Type) (borders : countries → set countries) (triangulated_sphere : Prop) :
  three_neighbor_countries countries borders triangulated_sphere →
  ¬ ∃ (v₁ v₂ : countries), v₁ ∈ countries ∧ v₂ ∈ countries ∧ ¬disjoint (borders v₁) (borders v₂) ∧
  (odd (card (borders v₁)) ∧ odd (card (borders v₂))) :=
begin
  sorry
end

end no_two_adjacent_odd_vertices_exist_l508_508675


namespace scientific_notation_of_11580000_l508_508925

theorem scientific_notation_of_11580000 :
  11_580_000 = 1.158 * 10^7 :=
sorry

end scientific_notation_of_11580000_l508_508925


namespace equation_of_line_with_x_intercept_and_slope_l508_508627

theorem equation_of_line_with_x_intercept_and_slope :
  ∃ (a b c : ℝ), a * x - b * y + c = 0 ∧ a = 1 ∧ b = -1 ∧ c = 2 :=
sorry

end equation_of_line_with_x_intercept_and_slope_l508_508627


namespace exponential_function_l508_508453

noncomputable def f : ℝ → ℝ

axiom condition1 : ∀ (x1 x2 : ℝ), f (x1 + x2) = f x1 * f x2
axiom condition2 : ∀ (x1 x2 : ℝ), x1 < x2 → f x1 < f x2

theorem exponential_function (a : ℝ) (h : a > 1) : ∃ f, (∀ (x : ℝ), f x = a^x) :=
sorry

end exponential_function_l508_508453


namespace mike_passing_percentage_l508_508240

theorem mike_passing_percentage (scored shortfall max_marks : ℝ) (total_marks := scored + shortfall) :
    scored = 212 →
    shortfall = 28 →
    max_marks = 800 →
    (total_marks / max_marks) * 100 = 30 :=
by
  intros
  sorry

end mike_passing_percentage_l508_508240


namespace measure_of_largest_angle_l508_508282

noncomputable def largest_angle_of_triangle (u : ℝ) (h1 : sqrt (3 * u - 2) + sqrt (3 * u + 2) > 2 * sqrt u)
  (h2 : sqrt (3 * u - 2) + 2 * sqrt u > sqrt (3 * u + 2))
  (h3 : 2 * sqrt u + sqrt (3 * u + 2) > sqrt (3 * u - 2)) : ℝ :=
  Real.arccos (u / Real.sqrt ((3 * u - 2) * (3 * u + 2)))

theorem measure_of_largest_angle (u : ℝ) (h1 : sqrt (3 * u - 2) + sqrt (3 * u + 2) > 2 * sqrt u)
  (h2 : sqrt (3 * u - 2) + 2 * sqrt u > sqrt (3 * u + 2))
  (h3 : 2 * sqrt u + sqrt (3 * u + 2) > sqrt (3 * u - 2)) :
  largest_angle_of_triangle u h1 h2 h3 = Real.arccos (u / Real.sqrt ((3 * u - 2) * (3 * u + 2))) :=
sorry

end measure_of_largest_angle_l508_508282


namespace line_equation_correct_l508_508162

def point := (ℝ × ℝ)

def inclination_angle : ℝ := 30
def point_on_line : point := (real.sqrt 3, -3)
def correct_line_equation (x : ℝ) : ℝ := (real.sqrt 3 / 3) * x - 4

theorem line_equation_correct :
  ∃ (m : ℝ) (b : ℝ), (m = real.sqrt 3 / 3 ∧ b = -4) ∧
  ∀ x, (x, correct_line_equation x) ∈ {p : point | p.2 = (real.sqrt 3 / 3) * p.1 + b} :=
by sorry

end line_equation_correct_l508_508162


namespace abs_sum_of_first_30_terms_l508_508492

def a : ℕ → ℤ
| 0       := -60
| (n + 1) := a n + 3

theorem abs_sum_of_first_30_terms : (Finset.range 30).sum (λ n, |a n|) = 765 :=
sorry

end abs_sum_of_first_30_terms_l508_508492


namespace complement_of_A_in_U_l508_508212

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {2, 4}

theorem complement_of_A_in_U : (U \ A) = {1, 3, 5} := by
  sorry

end complement_of_A_in_U_l508_508212


namespace female_guests_division_vegetarian_dietary_preferences_plus_one_arrangement_l508_508592

theorem female_guests_division (total_guests : ℕ) (percent_female : ℕ) (female_jay_family_percent : ℕ) :
  total_guests = 400 → 
  percent_female = 58 →
  female_jay_family_percent = 46 →
  let total_females := (percent_female * total_guests) / 100,
      females_jay := (female_jay_family_percent * total_females) / 100,
      females_mary := total_females - females_jay in
  females_jay = 107 ∧ females_mary = 125 := by
    sorry

theorem vegetarian_dietary_preferences (total_guests : ℕ) (percent_immediate_family : ℕ) (percent_friends : ℕ) (percent_extended_family : ℕ) (percent_vegetarian : ℕ) :
  total_guests = 400 →
  percent_immediate_family = 20 →
  percent_friends = 40 →
  percent_extended_family = 40 →
  percent_vegetarian = 35 →
  let immediate_family := (percent_immediate_family * total_guests) / 100,
      friends := (percent_friends * total_guests) / 100,
      extended_family := (percent_extended_family * total_guests) / 100,
      vegetarian_immediate_family := (percent_vegetarian * immediate_family) / 100,
      vegetarian_friends := (percent_vegetarian * friends) / 100,
      vegetarian_extended_family := (percent_vegetarian * extended_family) / 100 in
  vegetarian_immediate_family = 28 ∧ vegetarian_friends = 56 ∧ vegetarian_extended_family = 56 := by
    sorry

theorem plus_one_arrangement (total_guests : ℕ) (percent_plus_one : ℕ) (percent_seating_arrangement : ℕ) :
  total_guests = 400 →
  percent_plus_one = 15 →
  percent_seating_arrangement = 45 →
  let plus_one := (percent_plus_one * total_guests) / 100,
      specific_seating_arrangement := (percent_seating_arrangement * plus_one) / 100 in
  specific_seating_arrangement = 27 := by
    sorry

end female_guests_division_vegetarian_dietary_preferences_plus_one_arrangement_l508_508592


namespace power_log_simplification_l508_508321

theorem power_log_simplification :
  ((625 ^ (Real.log 2015 / Real.log 5)) ^ (1 / 4)) = 2015 :=
by
  sorry

end power_log_simplification_l508_508321


namespace correct_quadratic_equation_l508_508938

theorem correct_quadratic_equation :
  ∃ (b c : ℤ), (∀ (x : ℝ), (x - 5) * (x - 1) = x^2 + b * x + c) ∧ 
               (∀ (x : ℝ), (x + 7) * (x + 2) = x^2 + b * x + c) ∧ 
               (x^2 - 6 * x + 14 = 0) :=
begin
  sorry
end

end correct_quadratic_equation_l508_508938


namespace DF_perp_FG_l508_508941

-- Definitions of points and conditions for the Lean 4 statement
variable {A B C M D E F G N : Type}
variable [inhabited A] [inhabited B] [inhabited C]
variable [inhabited M] [inhabited D] [inhabited E]
variable [inhabited F] [inhabited G] [inhabited N]

-- Given conditions
variable (acute_triangle_ABC : ∀ {A B C : Type}, Prop)
variable (AB_less_AC : ∀ {A B C : Type}, Prop)
variable (M_midpoint_BC : ∀ {B C M : Type}, Prop)
variable (D_midpoint_arcBAC : ∀ {A B C D : Type}, Prop)
variable (E_midpoint_arcBC : ∀ {B C E : Type}, Prop)
variable (F_tangency_AB : ∀ {A B F : Type}, Prop)
variable (G_intersection_AE_BC : ∀ {A E G B C : Type}, Prop)
variable (N_on_EF_perp_AB : ∀ {E F N B A : Type}, Prop)
variable (BN_equals_EM : ∀ {B N E M : Type}, Prop)

-- The goal to prove
theorem DF_perp_FG :
  acute_triangle_ABC → AB_less_AC → M_midpoint_BC → D_midpoint_arcBAC → 
  E_midpoint_arcBC → F_tangency_AB → G_intersection_AE_BC → N_on_EF_perp_AB → 
  BN_equals_EM → (∀ {D F G : Type}, Prop) :=
by
  sorry

end DF_perp_FG_l508_508941


namespace blake_change_given_l508_508749

theorem blake_change_given :
  let oranges := 40
  let apples := 50
  let mangoes := 60
  let total_amount := 300
  let total_spent := oranges + apples + mangoes
  let change_given := total_amount - total_spent
  change_given = 150 :=
by
  sorry

end blake_change_given_l508_508749


namespace Patricia_earns_more_l508_508962

-- Define the function for compound interest with annual compounding
noncomputable def yearly_compound (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r)^n

-- Define the function for compound interest with quarterly compounding
noncomputable def quarterly_compound (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r / 4)^ (4 * n)

-- Define the conditions
def P := 50000.0
def r := 0.04
def n := 2

-- Define the amounts for Jose and Patricia using their respective compounding methods
def A_Jose := yearly_compound P r n
def A_Patricia := quarterly_compound P r n

-- Define the target difference in earnings
def difference := A_Patricia - A_Jose

-- Theorem statement
theorem Patricia_earns_more : difference = 63 := by
  sorry

end Patricia_earns_more_l508_508962


namespace product_floor_ceil_sequence_l508_508405

theorem product_floor_ceil_sequence :
  (Int.floor (-6 - 0.5) * Int.ceil (6 + 0.5) *
   Int.floor (-5 - 0.5) * Int.ceil (5 + 0.5) *
   Int.floor (-4 - 0.5) * Int.ceil (4 + 0.5) *
   Int.floor (-3 - 0.5) * Int.ceil (3 + 0.5) *
   Int.floor (-2 - 0.5) * Int.ceil (2 + 0.5) *
   Int.floor (-1 - 0.5) * Int.ceil (1 + 0.5) *
   Int.floor (-0.5) * Int.ceil (0.5)) = -25401600 := sorry

end product_floor_ceil_sequence_l508_508405


namespace find_tangency_segments_equal_l508_508013

-- Conditions of the problem as a theorem statement
theorem find_tangency_segments_equal (AB BC CD DA : ℝ) (x y : ℝ)
    (h1 : AB = 80)
    (h2 : BC = 140)
    (h3 : CD = 100)
    (h4 : DA = 120)
    (h5 : x + y = CD)
    (tangency_property : |x - y| = 0) :
  |x - y| = 0 :=
sorry

end find_tangency_segments_equal_l508_508013


namespace triangle_inequality_l508_508968

-- Definitions based on conditions
variables {A B C : Type} [triangle A B C]
variables {L : Type} [line L]
variables {u v w S : ℝ}

-- Assumptions
variable (h1 : ∀ {A B C : Type}, acute_triangle A B C)
variable (h2 : ∀ {A B C L : Type}, perpendicular_length A L = u)
variable (h3 : ∀ {A B L C : Type}, perpendicular_length B L = v)
variable (h4 : ∀ {A B C L : Type}, perpendicular_length C L = w)
variable (h5 : ∀ {A B C : Type}, area A B C = S)

-- Inequality to prove
theorem triangle_inequality :
  u ^ 2 * (tan (∡A)) + v ^ 2 * (tan (∡B)) + w ^ 2 * (tan (∡C)) ≥ 2 * S :=
sorry

end triangle_inequality_l508_508968


namespace martin_spends_30_dollars_on_berries_l508_508239

def cost_per_package : ℝ := 2.0
def cups_per_package : ℝ := 1.0
def cups_per_day : ℝ := 0.5
def days : ℝ := 30

theorem martin_spends_30_dollars_on_berries :
  (days / (cups_per_package / cups_per_day)) * cost_per_package = 30 :=
by
  sorry

end martin_spends_30_dollars_on_berries_l508_508239


namespace symmetry_axis_stretch_l508_508259

theorem symmetry_axis_stretch :
  let f (x : ℝ) := sin (x + π / 6)
  let g (x : ℝ) := sin (x / 2 + π / 6)
  (∃ k : ℤ, x = 2 * π / 3 + 2 * k * π) → x = 2 * π / 3 :=
by
  let f (x : ℝ) := sin (x + π / 6)
  let g (x : ℝ) := sin (x / 2 + π / 6)
  assume h : ∃ k : ℤ, x = 2 * π / 3 + 2 * k * π
  obtain ⟨k, hk⟩ := h
  sorry

end symmetry_axis_stretch_l508_508259


namespace probability_of_at_least_half_correct_l508_508990

open ProbabilityTheory

-- Define the conditions
def numQuestions : ℕ := 20
def successProbability : ℚ := 1 / 4
def requiredCorrectAnswers : ℕ := 10

-- Define the binomial probability function
noncomputable def binomialProbability (n k : ℕ) (p : ℚ) : ℚ :=
  nat.summation (λ i, if h : i ≤ k then nat.choose n i * p ^ i * (1 - p) ^ (n - i) else 0) (Finset.Icc 0 n)

-- Define the statement to be proved
theorem probability_of_at_least_half_correct :
  binomialProbability numQuestions (numQuestions - requiredCorrectAnswers) successProbability = 129 / 10000 :=
sorry

end probability_of_at_least_half_correct_l508_508990


namespace sum_of_negative_integers_abs_between_1_and_5_l508_508288

-- Definitions
def is_negative (x : ℤ) : Prop := x < 0
def abs_value_greater_than_one (x : ℤ) : Prop := Int.natAbs x > 1
def abs_value_less_than_five (x : ℤ) : Prop := Int.natAbs x < 5

-- The proof problem statement
theorem sum_of_negative_integers_abs_between_1_and_5 :
  (∑ x in {x : ℤ | is_negative x ∧ abs_value_greater_than_one x ∧ abs_value_less_than_five x}, x) = -9 :=
by {
  sorry
}

end sum_of_negative_integers_abs_between_1_and_5_l508_508288


namespace scientific_notation_11580000_l508_508918

theorem scientific_notation_11580000 :
  11580000 = 1.158 * 10^7 :=
sorry

end scientific_notation_11580000_l508_508918


namespace count_special_integers_l508_508505

theorem count_special_integers :
  (∃ n : ℕ, n = card { k ∈ finset.range 301 | k % lcm 4 6 = 0 ∧ k % 5 ≠ 0 ∧ k % 8 ≠ 0 }) ∧
  n = 10 :=
by
  sorry

end count_special_integers_l508_508505


namespace find_age_of_second_person_l508_508662

variable (T A X : ℝ)

def average_original_group (T A : ℝ) : Prop :=
  T = 7 * A

def average_with_39 (T A : ℝ) : Prop :=
  T + 39 = 8 * (A + 2)

def average_with_second_person (T A X : ℝ) : Prop :=
  T + X = 8 * (A - 1) 

theorem find_age_of_second_person (T A X : ℝ) 
  (h1 : average_original_group T A)
  (h2 : average_with_39 T A)
  (h3 : average_with_second_person T A X) :
  X = 15 :=
sorry

end find_age_of_second_person_l508_508662


namespace problem_statement_l508_508584

noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

theorem problem_statement :
  let A : ℝ × ℝ := (0, 7)
  let B : ℝ × ℝ := (0, 10)
  let C : ℝ × ℝ := (3, 6)
  let line_y_eq_x (P : ℝ × ℝ) := P.1 = P.2
  let intersects (L1 L2 : ℝ → ℝ) (P : ℝ × ℝ) := L1 P.1 = P.2 ∧ L2 P.1 = P.2
  ∃ A' B' : ℝ × ℝ,
    line_y_eq_x A' ∧
    line_y_eq_x B' ∧
    intersects (λ x, -1/3 * x + 7) (λ x, id x) A' ∧
    intersects (λ x, -4/3 * x + 10) (λ x, id x) B' ∧
    distance A' B' = 3 * real.sqrt 2 / 28 :=
by
  sorry

end problem_statement_l508_508584


namespace functional_equation_solution_l508_508076

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y z t : ℝ, (f x + f z) * (f y + f t) = f (x * y - z * t) + f (x * t + y * z)) →
  (f = 0 ∨ f = λ _, 1 / 2 ∨ f = λ x, x ^ 2) :=
by
  -- Proof goes here
  sorry

end functional_equation_solution_l508_508076


namespace joseph_baseball_cards_percentage_l508_508963

theorem joseph_baseball_cards_percentage :
  let original_cards := 24
  let brother_percentage := (1/3 : ℝ)
  let cousin_percentage := (1/4 : ℝ)
  let exchanged_given := 6
  let exchanged_gained := 4

  let after_brother := original_cards * (1 - brother_percentage)
  let after_cousin := after_brother * (1 - cousin_percentage)
  let after_exchange := after_cousin - exchanged_given + exchanged_gained

  let percentage_left := (after_exchange / original_cards) * 100
  percentage_left ≈ 41.67 :=
by
  sorry

end joseph_baseball_cards_percentage_l508_508963


namespace restore_supply_function_determine_tax_revenue_determine_optimal_tax_rate_determine_maximum_tax_revenue_l508_508365

-- Defining the given conditions
noncomputable def market_demand (P : ℝ) : ℝ := 688 - 4 * P
noncomputable def post_tax_producer_price : ℝ := 64
noncomputable def per_unit_tax : ℝ := 90
noncomputable def elasticity_supply_no_tax (P_e : ℝ) (Q_e : ℝ) : ℝ :=
  1.5 * (-(4 * P_e / Q_e))

-- Supply function to be proven
noncomputable def supply_function (P : ℝ) : ℝ := 6 * P - 312

-- Total tax revenue to be proven
noncomputable def total_tax_revenue : ℝ := 6480

-- Optimal tax rate to be proven
noncomputable def optimal_tax_rate : ℝ := 60

-- Maximum tax revenue to be proven
noncomputable def maximum_tax_revenue : ℝ := 8640

-- Theorem statements that need to be proven
theorem restore_supply_function (P : ℝ) : 
  supply_function P = 6 * P - 312 := sorry

theorem determine_tax_revenue : 
  total_tax_revenue = 6480 := sorry

theorem determine_optimal_tax_rate : 
  optimal_tax_rate = 60 := sorry

theorem determine_maximum_tax_revenue : 
  maximum_tax_revenue = 8640 := sorry

end restore_supply_function_determine_tax_revenue_determine_optimal_tax_rate_determine_maximum_tax_revenue_l508_508365


namespace dot_product_example_l508_508808

def vec_add (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)

def dot_product (u v : ℝ × ℝ) : ℝ := (u.1 * v.1 + u.2 * v.2)

theorem dot_product_example :
  let a := (1, -2) in
  let b := (3, 4) in
  let c := (2, -1) in
  dot_product (vec_add a b) c = 6 :=
by
  sorry

end dot_product_example_l508_508808


namespace scientific_notation_11580000_l508_508917

theorem scientific_notation_11580000 :
  11580000 = 1.158 * 10^7 :=
sorry

end scientific_notation_11580000_l508_508917


namespace min_squares_to_cover_5x5_l508_508250

theorem min_squares_to_cover_5x5 : 
  (∀ (cover : ℕ → ℕ), (cover 1 + cover 2 + cover 3 + cover 4) * (1^2 + 2^2 + 3^2 + 4^2) = 25 → 
  cover 1 + cover 2 + cover 3 + cover 4 = 10) :=
sorry

end min_squares_to_cover_5x5_l508_508250


namespace students_in_5th_6th_grades_l508_508281

-- Definitions for problem conditions
def is_three_digit_number (n : ℕ) : Prop :=
  n >= 100 ∧ n < 1000

def six_two_digit_sum_eq_twice (n : ℕ) : Prop :=
  ∃ a b c : ℕ, n = 100 * a + 10 * b + c ∧
               a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
               (10 * a + b) + (10 * b + a) + (10 * a + c) + (10 * c + a) + (10 * b + c) + (10 * c + b) = 2 * n

-- The proof problem statement in Lean 4
theorem students_in_5th_6th_grades :
  ∃ n : ℕ, is_three_digit_number n ∧ six_two_digit_sum_eq_twice n ∧ n = 198 :=
by
  sorry

end students_in_5th_6th_grades_l508_508281


namespace gcd_840_1764_l508_508633

theorem gcd_840_1764 : gcd 840 1764 = 84 := 
by
  -- proof steps will go here
  sorry

end gcd_840_1764_l508_508633


namespace total_cases_of_cat_food_sold_l508_508297

theorem total_cases_of_cat_food_sold :
  (let first_eight := 8 * 3 in
   let next_four := 4 * 2 in
   let last_eight := 8 * 1 in
   first_eight + next_four + last_eight = 40) :=
by
  -- Given conditions:
  -- first_8_customers: 8 customers bought 3 cases each
  -- second_4_customers: 4 customers bought 2 cases each
  -- last_8_customers: 8 customers bought 1 case each
  let first_eight := 8 * 3
  let next_four := 4 * 2
  let last_eight := 8 * 1
  -- Sum of all cases
  show first_eight + next_four + last_eight = 40
  sorry

end total_cases_of_cat_food_sold_l508_508297


namespace composite_even_of_even_l508_508983

variable {α : Type*} [AddGroup α] (g : α → α)

-- The condition that g is even.
def is_even_function (g : α → α) : Prop :=
∀ x : α, g (-x) = g x

-- The proof goal that g(g(x)) is even.
theorem composite_even_of_even (h : is_even_function g) : is_even_function (g ∘ g) :=
by
  intro x
  calc
    (g ∘ g) (-x) = g (g (-x)) : rfl
              ... = g (g x) : by rw [h]
              ... = (g ∘ g) (x) : rfl

end composite_even_of_even_l508_508983


namespace tan_product_alpha_beta_tan_product_1_to_45_l508_508509

theorem tan_product_alpha_beta (α β : ℝ) (hα : 0 < α) (hβ : 0 < β) (hαβ : α + β = Real.pi / 4) :
    (1 + Real.tan α) * (1 + Real.tan β) = 2 := 
sorry

theorem tan_product_1_to_45 :
    ∏ k in Finset.range 45, (1 + Real.tan (↑k + 1 : ℕ) * Real.pi / 180) = 2 ^ 23 := 
sorry

end tan_product_alpha_beta_tan_product_1_to_45_l508_508509


namespace find_coordinates_of_C_l508_508827

structure Point :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def A : Point := { x := 4, y := -1, z := 2 }
def B : Point := { x := 2, y := -3, z := 0 }

def satisfies_condition (C : Point) : Prop :=
  (C.x - B.x, C.y - B.y, C.z - B.z) = (2 * (A.x - C.x), 2 * (A.y - C.y), 2 * (A.z - C.z))

theorem find_coordinates_of_C (C : Point) (h : satisfies_condition C) : C = { x := 10/3, y := -5/3, z := 4/3 } :=
  sorry -- Proof is omitted as requested

end find_coordinates_of_C_l508_508827


namespace factor_poly_find_abs_l508_508172

theorem factor_poly_find_abs {
  p q : ℤ
} (h1 : 3 * (-2)^3 - p * (-2) + q = 0) 
  (h2 : 3 * (3)^3 - p * (3) + q = 0) :
  |3 * p - 2 * q| = 99 := sorry

end factor_poly_find_abs_l508_508172


namespace first_question_second_question_l508_508557

-- Define the 101-gon and its vertices
def regular_101gon := set (fin 101 → Prop)

-- Define the coloring conditions
def is_red (v : fin 101) : bool := sorry
def is_blue (v : fin 101) : bool := not (is_red v)

-- Define the conditions for obtuse triangles
def obtuse_triangle (i j k : fin 101) : Prop := sorry
def acute_angle_same_color (i j k : fin 101) : Prop := sorry
def obtuse_angle_different_color (i j k : fin 101) : Prop := sorry

-- Number of valid obtuse triangles
def N : ℕ := sorry

-- The maximum number of valid triangles
def max_N : ℕ := 32175

-- Number of ways to color the vertices
def valid_colorings := 202 * (nat.choose 75 25)

-- The proof statements
theorem first_question : N = max_N := sorry
theorem second_question : (number_of_ways_to_color := 202 * nat.choose 75 25 := sorry

end first_question_second_question_l508_508557


namespace solve_inequality_l508_508481

def f (a x : ℝ) : ℝ := a * x * (x + 1) + 1

theorem solve_inequality (a x : ℝ) (h : f a x < 0) : x < (1 / a) ∨ (x > 1 ∧ a ≠ 0) := by
  sorry

end solve_inequality_l508_508481


namespace solve_for_y_l508_508043

theorem solve_for_y {y : ℝ} : 
  (2012 + y)^2 = 2 * y^2 ↔ y = 2012 * (Real.sqrt 2 + 1) ∨ y = -2012 * (Real.sqrt 2 - 1) := by
  sorry

end solve_for_y_l508_508043


namespace abie_gave_4_bags_l508_508718

-- Definitions based on the conditions
def initial_bags : ℕ := 20
def bought_bags : ℕ := 6
def final_bags : ℕ := 22

-- The number of bags Abie gave to her friend
def given_bags : ℕ → Prop := λ x, initial_bags - x + bought_bags = final_bags

-- The proof statement
theorem abie_gave_4_bags : given_bags 4 :=
by
  -- Skipping the proof as instructed
  sorry

end abie_gave_4_bags_l508_508718


namespace smallest_b_exists_l508_508798

theorem smallest_b_exists :
  ∃ b : ℕ, (∀ r s : ℤ, r * s = 4032 ∧ r + s = b) ∧
    (∀ b' : ℕ, (∀ r' s' : ℤ, r' * s' = 4032 ∧ r' + s' = b') → b ≤ b') :=
sorry

end smallest_b_exists_l508_508798


namespace find_f_val_l508_508849

noncomputable def f (x : ℝ) (a b c : ℝ) : ℝ :=
  a * (Real.tan x) ^ 2017 + b * x ^ 2017 + c * Real.log (x + Real.sqrt (x ^ 2 + 1)) + 20

theorem find_f_val (a b c : ℝ) (h : f (Real.log (Real.log (21) / Real.log(2))) a b c = 17) :
  f (Real.log (Real.log(5) / Real.log(21))) a b c = 23 :=
sorry

end find_f_val_l508_508849


namespace monotonicity_and_range_l508_508853

noncomputable def f (a x : ℝ) : ℝ := (a * x - 2) * Real.exp x - Real.exp (a - 2)

theorem monotonicity_and_range (a x : ℝ) :
  ( (a = 0 → ∀ x, f a x < f a (x + 1)) ∧
  (a > 0 → ∀ x < (2 - a) / a, f a x < f a (x + 1) ∧ ∀ x > (2 - a) / a, f a x > f a (x + 1) ) ∧
  (a < 0 → ∀ x > (2 - a) / a, f a x < f a (x + 1) ∧ ∀ x < (2 - a) / a, f a x > f a (x + 1) ) ∧
  (∀ x > 1, f a x > 0 ↔ a ∈ Set.Ici 1)) 
:=
sorry

end monotonicity_and_range_l508_508853


namespace profit_percentage_correct_l508_508728

noncomputable def cost_price : ℝ := 47.50
noncomputable def selling_price : ℝ := 70
noncomputable def list_price : ℝ := selling_price / 0.95
noncomputable def profit : ℝ := selling_price - cost_price
noncomputable def profit_percentage : ℝ := (profit / cost_price) * 100

theorem profit_percentage_correct :
  abs (profit_percentage - 47.37) < 0.01 := sorry

end profit_percentage_correct_l508_508728


namespace sum_2001_and_1015_l508_508320

theorem sum_2001_and_1015 :
  2001 + 1015 = 3016 :=
sorry

end sum_2001_and_1015_l508_508320


namespace ball_placement_count_l508_508444

-- Definitions for the balls and their numbering
inductive Ball
| b1
| b2
| b3
| b4

-- Definitions for the boxes and their numbering
inductive Box
| box1
| box2
| box3

-- Function that checks if an assignment is valid given the conditions.
def isValidAssignment (assignment : Ball → Box) : Prop :=
  assignment Ball.b1 ≠ Box.box1 ∧ assignment Ball.b3 ≠ Box.box3

-- Main statement to prove
theorem ball_placement_count : 
  ∃ (assignments : Finset (Ball → Box)), 
    (∀ f ∈ assignments, isValidAssignment f) ∧ assignments.card = 14 := 
sorry

end ball_placement_count_l508_508444


namespace line_through_chord_with_midpoint_l508_508480

theorem line_through_chord_with_midpoint (x y : ℝ) :
  (∃ x1 y1 x2 y2 : ℝ,
    (x = x1 ∧ y = y1 ∨ x = x2 ∧ y = y2) ∧
    x = -1 ∧ y = 1 ∧
    x1^2 / 4 + y1^2 / 3 = 1 ∧
    x2^2 / 4 + y2^2 / 3 = 1) →
  3 * x - 4 * y + 7 = 0 :=
by
  sorry

end line_through_chord_with_midpoint_l508_508480


namespace AP_eq_PQ_l508_508205

-- Definitions for points and triangle elements
variable (A B C P Q : Type) [incidence_geometry A B C P Q]

def is_tangent_at (circle : Type) (A : Type) (tangent : Type) : Prop := sorry
def circumcircle (triangle : Type) : Type := sorry
def angle_bisector (A B : Type) : Type := sorry
def intersection (line1 line2 : Type) : Type := sorry

-- Conditions
axiom triangle_ABC : Triangle A B C
axiom circumcircle_ABC : Circle A (= circumcircle triangle_ABC)
axiom tangent_at_A : is_tangent_at circumcircle_ABC A BC
axiom intersection_tangent_BC_at_P : P = intersection (tangent_at_A) BC
axiom intersection_angle_bisector_BC_at_Q : Q = intersection (angle_bisector A B) BC

-- Goal
theorem AP_eq_PQ : length (A - P) = length (P - Q) := 
by sorry

end AP_eq_PQ_l508_508205


namespace count_special_integers_l508_508504

theorem count_special_integers :
  (∃ n : ℕ, n = card { k ∈ finset.range 301 | k % lcm 4 6 = 0 ∧ k % 5 ≠ 0 ∧ k % 8 ≠ 0 }) ∧
  n = 10 :=
by
  sorry

end count_special_integers_l508_508504


namespace solve_mod_equiv_l508_508656

theorem solve_mod_equiv : ∃ (n : ℤ), 0 ≤ n ∧ n < 9 ∧ (-2222 ≡ n [ZMOD 9]) → n = 6 := by
  sorry

end solve_mod_equiv_l508_508656


namespace scale_model_height_l508_508004

-- Define the actual height of the Eiffel Tower in feet
def actual_height : ℝ := 1063

-- Define the scale ratio
def scale_ratio : ℝ := 50

-- Define conversion factor from feet to inches
def feet_to_inches : ℝ := 12

-- Define the expected height of the scale model in inches
def expected_height_in_inches : ℕ := 255

-- Prove the height of the scale model in inches
theorem scale_model_height :
  (actual_height / scale_ratio * feet_to_inches).round = expected_height_in_inches :=
by
  sorry

end scale_model_height_l508_508004


namespace general_formula_sum_first_n_b_l508_508116

open Nat

variables (a : ℕ → ℤ) (b : ℕ → ℤ) (T : ℕ → ℤ)
variables (d : ℤ := 3)

-- conditions
def is_arithmetic_seq (a : ℕ → ℤ) (d : ℤ) := ∀ n, a (n + 1) = a n + d
def is_geometric_seq (a : ℕ → ℤ) := a 3 ^ 2 = a 1 * a 11

-- Given conditions
def a_1_eq_2 : a 1 = 2 := by sorry
def geometric_seq : is_geometric_seq a := by sorry

-- Problem restatement in Lean
theorem general_formula (h1 : is_arithmetic_seq a d) (h2 : a 1 = 2) (h3 : is_geometric_seq a) :
  ∀ n, a n = 3 * n - 1 := 
by sorry

-- Define the summation b_n and T_n
def b (n : ℕ) : ℤ := a n - 2 ^ n - 3 / 2
def sum_b (n : ℕ) : ℤ := (finset.range n).sum (λ k, b (k + 1))

-- Given the general formula for a_n
def a_formula : ∀ n, a n = 3 * n - 1 := by sorry

-- Final part to prove
theorem sum_first_n_b (h1 : is_arithmetic_seq a d) (h2 : a 1 = 2) (h3 : is_geometric_seq a) :
  ∀ n, sum_b n = (3 * n ^ 2 / 2) - 2 ^ (n + 1) + 2 := 
by sorry

end general_formula_sum_first_n_b_l508_508116


namespace parrots_per_cage_l508_508705

-- Definitions of the given conditions
def num_cages : ℕ := 6
def num_parakeets_per_cage : ℕ := 7
def total_birds : ℕ := 54

-- Proposition stating the question and the correct answer
theorem parrots_per_cage : (total_birds - num_cages * num_parakeets_per_cage) / num_cages = 2 := 
by
  sorry

end parrots_per_cage_l508_508705


namespace find_f_55_l508_508569

noncomputable def f (x : ℤ) : ℤ := sorry

axiom f_1 : f 1 = 1
axiom f_2 : f 2 = 0
axiom f_neg1 : f (-1) < 0
axiom functional_equation : ∀ x y : ℤ, f(x + y) = f(x) * f(1 - y) + f(1 - x) * f(y)

theorem find_f_55 : f 55 = -1 :=
by sorry

end find_f_55_l508_508569


namespace parallelogram_perimeter_l508_508796

-- Define the given conditions
variables {A B C D : Type}
variables (AB BC AD CD : ℝ) (AC BD : A)

-- Given properties
def is_parallelogram (AB BC AD CD : ℝ) : Prop := (AB = 35) ∧ (BC = 42) ∧ (AD = 35) ∧ (CD = 42)
def diagonals_intersect_right_angle (AC BD : A) : Prop := true  -- This is symbolic and not specified further.

-- Proof statement
theorem parallelogram_perimeter (hP : is_parallelogram AB BC AD CD) (hD : diagonals_intersect_right_angle AC BD) : (AB + BC + CD + AD = 154) :=
by {
  -- Unpack the conditions from hP
  cases hP with h1 h2,
  cases h2 with h3 h4,

  -- Use the provided conditions
  rw [h1, h3, h2.1, h2.2],

  -- Directly compute the perimeter using the sums
  norm_num,
}

end parallelogram_perimeter_l508_508796


namespace area_of_lune_l508_508711

/-- The area of the lune formed by two semicircles, one with radius 3 cm, 
and another with diameter 8 cm, is 7/2 * pi cm². --/
theorem area_of_lune (r1 r2 : ℝ) (π : ℝ) (h1 : r1 = 3) (h2 : 2 * r2 = 8) : 
  (1 / 2 * π * r2^2) - (1 / 2 * π * r1^2) = 7 / 2 * π :=
begin
  rw [h1, h2],
  sorry
end

end area_of_lune_l508_508711


namespace associations_same_members_l508_508932

-- Definitions of the conditions
variable (V : Type) (inhabitants : set V) (associations : set (set V))
variable (k : Nat)
variable [nonempty inhabitants]

-- Assumptions
variable (h1 : ∀ v ∈ inhabitants, ∃ a ⊆ associations, a.card ≥ k ∧ ∀ x y ∈ a, x ≠ y → (x ∩ y).card ≤ 1)
variable (h2 : ∃ x ∈ inhabitants)

-- The main theorem to prove
theorem associations_same_members (h1 : ∀ v ∈ inhabitants, ∃ a ⊆ associations, a.card ≥ k ∧ ∀ x y ∈ a, x ≠ y → (x ∩ y).card ≤ 1)
(h2 : ∃ x ∈ inhabitants) : ∃ A ⊆ associations, A.card ≥ k ∧ ∃ n : ℕ, ∀ B ∈ A, B.card = n := 
sorry

end associations_same_members_l508_508932


namespace percentage_increase_overtime_l508_508991

def regular_hours : ℕ := 20
def overtime_hours := 70 - regular_hours
def regular_rate : ℝ := 8
def max_earnings : ℝ := 660
def max_additional_earnings := max_earnings - (regular_hours * regular_rate)
def overtime_rate : ℝ := max_additional_earnings / overtime_hours
def percentage_increase := ((overtime_rate - regular_rate) / regular_rate) * 100

theorem percentage_increase_overtime :
  percentage_increase = 25 :=
by
  sorry

end percentage_increase_overtime_l508_508991


namespace total_cost_after_discount_l508_508197

def small_men_white_cost : ℕ := 20
def medium_men_white_cost : ℕ := 24
def large_men_white_cost : ℕ := 28
def small_men_black_cost : ℕ := 18
def medium_men_black_cost : ℕ := 22
def large_men_black_cost : ℕ := 26

def small_women_white_cost : ℕ := small_men_white_cost - 5
def medium_women_white_cost : ℕ := medium_men_white_cost - 5
def large_women_white_cost : ℕ := large_men_white_cost - 5
def small_women_black_cost : ℕ := small_men_black_cost - 5
def medium_women_black_cost : ℕ := medium_men_black_cost - 5
def large_women_black_cost : ℕ := large_men_black_cost - 5

def total_employees : ℕ := 60
def total_sectors : ℕ := 2
def total_tshirts : ℕ := total_employees * total_sectors
def small_percentage : ℝ := 0.5
def medium_percentage : ℝ := 0.3
def large_percentage : ℝ := 0.2

noncomputable def small_tshirts : ℕ := (small_percentage * total_tshirts).toInt
noncomputable def medium_tshirts : ℕ := (medium_percentage * total_tshirts).toInt
noncomputable def large_tshirts : ℕ := (large_percentage * total_tshirts).toInt

noncomputable def cost_before_discount : ℝ :=
  (small_men_white_cost + small_women_white_cost + small_men_black_cost + small_women_black_cost) * small_tshirts +
  (medium_men_white_cost + medium_women_white_cost + medium_men_black_cost + medium_women_black_cost) * medium_tshirts +
  (large_men_white_cost + large_women_white_cost + large_men_black_cost + large_women_black_cost) * large_tshirts

def discount_rate : ℝ := if total_tshirts > 100 then 0.1 else if total_tshirts > 50 then 0.05 else 0
noncomputable def discount_amount : ℝ := discount_rate * cost_before_discount
noncomputable def cost_after_discount : ℝ := cost_before_discount - discount_amount

theorem total_cost_after_discount : cost_after_discount = 8337.60 :=
by
  sorry

end total_cost_after_discount_l508_508197


namespace logarithmic_properties_l508_508839

-- Define the conditions as variables
variables (a b : ℝ)
variable (log_2_3 : a = Real.log 3 / Real.log 2)
variable (log_3_8 : b = Real.log 8 / Real.log 3)

-- The theorem to be proved
theorem logarithmic_properties (a b : ℝ) (log_2_3 : a = Real.log 3 / Real.log 2) (log_3_8 : b = Real.log 8 / Real.log 3) : 
  2 ^ a = 3 ∧ a * b = 3 := 
by {
  sorry
}

end logarithmic_properties_l508_508839


namespace smallest_even_piece_to_stop_triangle_l508_508025

-- Define a predicate to check if an integer is even
def even (x : ℕ) : Prop := x % 2 = 0

-- Define the conditions for triangle inequality to hold
def triangle_inequality_violated (a b c : ℕ) : Prop :=
  a + b ≤ c ∨ a + c ≤ b ∨ b + c ≤ a

-- Define the main theorem
theorem smallest_even_piece_to_stop_triangle
  (x : ℕ) (hx : even x) (len1 len2 len3 : ℕ)
  (h_len1 : len1 = 7) (h_len2 : len2 = 24) (h_len3 : len3 = 25) :
  6 ≤ x → triangle_inequality_violated (len1 - x) (len2 - x) (len3 - x) :=
by
  sorry

end smallest_even_piece_to_stop_triangle_l508_508025


namespace acute_angle_10_10_l508_508317

noncomputable def clock_angle_proof : Prop :=
  let minute_hand_position := 60
  let hour_hand_position := 305
  let angle_diff := hour_hand_position - minute_hand_position
  let acute_angle := if angle_diff > 180 then 360 - angle_diff else angle_diff
  acute_angle = 115

theorem acute_angle_10_10 : clock_angle_proof := by
  sorry

end acute_angle_10_10_l508_508317


namespace blake_change_l508_508746

def cost_oranges : ℕ := 40
def cost_apples : ℕ := 50
def cost_mangoes : ℕ := 60
def initial_money : ℕ := 300

def total_cost : ℕ := cost_oranges + cost_apples + cost_mangoes
def change : ℕ := initial_money - total_cost

theorem blake_change : change = 150 := by
  sorry

end blake_change_l508_508746


namespace count_valid_c_values_for_equation_l508_508805

theorem count_valid_c_values_for_equation :
  ∃ (c_set : Finset ℕ), (∀ c ∈ c_set, c ≤ 2000 ∧
    ∃ x : ℝ, 5 * (Real.floor x) + 3 * (Real.ceil x) = c) ∧
    c_set.card = 500 :=
by
  sorry

end count_valid_c_values_for_equation_l508_508805


namespace smallest_consecutive_odd_sum_l508_508283

theorem smallest_consecutive_odd_sum (a b c d e : ℤ)
    (h1 : b = a + 2)
    (h2 : c = a + 4)
    (h3 : d = a + 6)
    (h4 : e = a + 8)
    (h5 : a + b + c + d + e = 375) : a = 71 :=
by
  -- the proof will go here
  sorry

end smallest_consecutive_odd_sum_l508_508283


namespace jenna_water_cups_l508_508553

theorem jenna_water_cups (O S W : ℕ) (h1 : S = 3 * O) (h2 : W = 3 * S) (h3 : O = 4) : W = 36 :=
by
  sorry

end jenna_water_cups_l508_508553


namespace path_length_of_dot_l508_508017

-- Define the dimensions and rolling properties
structure Prism :=
  (length : ℕ)
  (width : ℕ)
  (height : ℕ)

structure RollState :=
  (dot_position : ℤ × ℤ)
  (revolutions : ℕ)

def initial_prism : Prism := ⟨2, 1, 1⟩

-- Define the conditions
def dot_initial_position : (ℤ × ℤ) := (1, 0)
def total_revolutions : ℕ := 2

-- Prove the path length of the dot
theorem path_length_of_dot (p : Prism) (s : RollState) (r : ℕ) (initial_pos : (ℤ × ℤ)) :
  p = initial_prism →
  s = { dot_position := initial_pos, revolutions := r } →
  initial_pos = dot_initial_position →
  r = total_revolutions →
  ∃ k : ℝ, k = 2 * Real.sqrt 5 ∧ s.dot_position = initial_pos :=
by {
  intros h1 h2 h3 h4,
  use 2 * Real.sqrt 5,
  split,
  { refl },        -- k = 2 * sqrt 5
  { simp [h3] }    -- dot_position is unchanged
}

end path_length_of_dot_l508_508017


namespace floor_ceil_product_l508_508408

theorem floor_ceil_product :
  let f : ℕ → ℤ := λ n, (Int.floor (- (n : ℤ) - 0.5)) * (Int.ceil ((n : ℤ) + 0.5))
  let product : ℤ := ∏ i in Finset.range 7, -(i + 1)^2
  ∑ n in Finset.range 7, f n = product :=
by
  sorry

end floor_ceil_product_l508_508408


namespace area_of_triangle_AHF_l508_508006

-- Definitions for the geometric setting
def parabola (x y : ℝ) : Prop := x^2 = 4 * y

def focus : ℝ × ℝ := (0, 1)

def line_through_focus (m : ℝ) (p : ℝ × ℝ) : Prop :=
  ∃ (b : ℝ), p.2 = m * p.1 + b

def intersects_at_point (m : ℝ) (p₁ p₂ : ℝ × ℝ) : Prop :=
  parabola p₂.1 p₂.2 ∧ (p₁ = focus) ∧ line_through_focus m p₁ ∧ p₂.2 = m * p₂.1 + 1

def perpendicular_to_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (0, p.2)

def area_of_triangle (a b c : ℝ × ℝ) : ℝ :=
  (1 / 2) * abs ((a.1 - c.1) * (b.2 - a.2) - (a.1 - b.1) * (c.2 - a.2))

-- The theorem to prove
theorem area_of_triangle_AHF :
  ∃ A : ℝ × ℝ, intersects_at_point (sqrt 3 / 3) focus A →
  let H := perpendicular_to_axis A in
  area_of_triangle A H focus = 4 * sqrt 3 :=
  sorry

end area_of_triangle_AHF_l508_508006


namespace solution_set_of_inequality_l508_508813

-- Define the given function and conditions
variable {f : ℝ → ℝ}
variable (hf1 : ∀ x : ℝ, x ∈ (-2, 2) → f(x) + exp(4 * x) * f(-x) = 0)
variable (hf2 : f(1) = exp 2)
variable (hf3 : ∀ x : ℝ, x ∈ [0, 2) → deriv f x > 2 * f(x))

-- Define the inequality
def satisfies_inequality (x : ℝ) := exp(2 * x) * f(2 - x) < exp(4)

-- The theorem to prove
theorem solution_set_of_inequality : {x : ℝ | x ∈ (1, 4) → satisfies_inequality x} :=
by
  sorry

end solution_set_of_inequality_l508_508813


namespace find_a_plus_b_l508_508568

noncomputable def f (a b x : ℝ) := a * x + b
noncomputable def g (x : ℝ) := 3 * x - 4

theorem find_a_plus_b (a b : ℝ) (h : ∀ (x : ℝ), g (f a b x) = 4 * x + 5) : a + b = 13 / 3 := 
  sorry

end find_a_plus_b_l508_508568


namespace problem2024_l508_508053

theorem problem2024 :
  (∑ k in Finset.range 2023 | (2024 - k) / (k + 1)) / (∑ k in (Finset.range 2023) + 1 / (k + 2)) = 2024 := sorry

end problem2024_l508_508053


namespace floor_area_of_ring_l508_508066

theorem floor_area_of_ring :
  (let R := 40 in
   let r := R / 3 in
   let M := Real.pi * R^2 - 8 * Real.pi * r^2 in
   ⌊ M ⌋ = 5268) :=
by {
  let R := 40,
  let r := R / 3,
  let M := Real.pi * R^2 - 8 * Real.pi * r^2,
  have := Real.floor (Real.pi * (40^2 - 8 * (40/3)^2)),
  exact this,
  sorry
}

end floor_area_of_ring_l508_508066


namespace min_steps_to_identical_l508_508937

theorem min_steps_to_identical :
  let words := ["ЗАМОЗА", "ЗКНОЗА", "ЗМОМАА", "ЗМОМОЗ", "ЗМОЗОЗ"]
  in min_steps_replace_letters words = 11 :=
sorry

end min_steps_to_identical_l508_508937


namespace total_distance_between_alice_bob_l508_508381

-- Define the constants for Alice's and Bob's speeds and the time duration in terms of conditions.
def alice_speed := 1 / 12  -- miles per minute
def bob_speed := 3 / 20    -- miles per minute
def time_duration := 120   -- minutes

-- Statement: Prove that the total distance between Alice and Bob after 2 hours is 28 miles.
theorem total_distance_between_alice_bob : (alice_speed * time_duration) + (bob_speed * time_duration) = 28 :=
by
  sorry

end total_distance_between_alice_bob_l508_508381


namespace range_of_z_l508_508615

theorem range_of_z (x y : ℝ) (h1 : |x + y| ≤ 2) (h2 : |x - y| ≤ 2) :
  ∀ (z : ℝ), z = y / (x - 4) → z ∈ Icc (-1 / 2) (1 / 2) :=
sorry

end range_of_z_l508_508615


namespace absolute_sum_sequence_l508_508099

noncomputable def a : ℕ → ℤ
| 0         := 0   -- dummy value because sequence starts from n = 1
| 1         := -20
| (n + 2)   := a (n + 1) + 2

theorem absolute_sum_sequence : (∑ i in Finset.range 21, |a i|.natAbs) = 200 := by
  sorry

end absolute_sum_sequence_l508_508099


namespace sqrt_problem_quadratic_solutions_l508_508677

-- (1) Prove that √18 - √24 / √3 = √2
theorem sqrt_problem : (sqrt 18 - sqrt 24 / sqrt 3) = sqrt 2 := 
sorry

-- (2) Prove that the solutions to the equation x^2 - 4x - 5 = 0 are x = 5 or x = -1
theorem quadratic_solutions (x : ℝ) : (x^2 - 4 * x - 5 = 0) ↔ (x = 5 ∨ x = -1) := 
sorry

end sqrt_problem_quadratic_solutions_l508_508677


namespace exists_positive_integer_N_l508_508973

def sequence_of_integer_vectors (A : ℕ → ℤ × ℤ) : Prop :=
  ∀ n : ℕ, ∃ (a b : ℤ), A n = (a, b)

theorem exists_positive_integer_N (A : ℕ → ℤ × ℤ) (h : sequence_of_integer_vectors A) :
  ∃ N : ℕ, 0 < N ∧ ∀ n : ℕ, ∃ (k : fin N → ℤ), A n = finset.univ.sum (λ i, k i • A i) :=
sorry

end exists_positive_integer_N_l508_508973


namespace cds_per_rack_l508_508064

theorem cds_per_rack (total_cds : ℕ) (racks_per_shelf : ℕ) (cds_per_rack : ℕ) 
  (h1 : total_cds = 32) 
  (h2 : racks_per_shelf = 4) : 
  cds_per_rack = total_cds / racks_per_shelf :=
by 
  sorry

end cds_per_rack_l508_508064


namespace arithmetic_seq_general_term_a_sum_S_l508_508493

-- Define the sequence {a_n} such that it satisfies the given conditions
def a : ℕ → ℝ
| n => (1 / (2 * (n + 1) + 7))

def b (n : ℕ) : ℝ := a n * a (n + 1)
def S (n : ℕ) : ℝ := (1 / 2) * (1 / 9 - 1 / (2 * (n + 1) + 9))

theorem arithmetic_seq (n : ℕ) : (1 : ℝ) / a (n + 1) - (1 : ℝ) / a n = 2 :=
  sorry

theorem general_term_a (n : ℕ) : a n = 1 / (2 * (n + 1) + 7) :=
  sorry

theorem sum_S (n : ℕ) : S n = n / (9 * (2 * (n + 1) + 9)) :=
  sorry

-- Given the conditions
axiom seq_condition (n : ℕ) : a n - a (n + 1) = 2 * a n * a (n + 1)

axiom initial_sum_condition : (1 / a 0 + 1 / a 1 + 1 / a 2 + 1 / a 3 + 1 / a 4) = 65

end arithmetic_seq_general_term_a_sum_S_l508_508493


namespace candies_distribution_l508_508994

theorem candies_distribution (C : ℕ) (hC : C / 150 = C / 300 + 24) : C / 150 = 48 :=
by sorry

end candies_distribution_l508_508994


namespace infinitely_many_k_composite_l508_508062

theorem infinitely_many_k_composite :
  ∃ᶠ k : ℕ in at_top, ∀ n : ℕ, 0 < n → ∃ p : ℕ, p.prime ∧ p ∣ (k * 2^n + 1) ∧ p < k * 2^n + 1 :=
sorry

end infinitely_many_k_composite_l508_508062


namespace angle_between_a_and_b_is_135_l508_508173

variables (a b : ℝ × ℝ)
def angle_between_vectors (u v : ℝ × ℝ) : ℝ :=
  real.arccos ((u.1 * v.1 + u.2 * v.2) / (real.sqrt (u.1^2 + u.2^2) * real.sqrt (v.1^2 + v.2^2)))

theorem angle_between_a_and_b_is_135 :
  a = (1, 2) →
  a + b = (2, -1) →
  angle_between_vectors a b = 3 * real.pi / 4 :=
by
  sorry

end angle_between_a_and_b_is_135_l508_508173


namespace arrange_exponents_l508_508447

noncomputable def a : ℝ := 0.8 ^ 0.7
noncomputable def b : ℝ := 0.8 ^ 0.9
noncomputable def c : ℝ := 1.2 ^ 0.8

theorem arrange_exponents (h1 : a = 0.8 ^ 0.7) (h2 : b = 0.8 ^ 0.9) (h3 : c = 1.2 ^ 0.8) : b < a ∧ a < c :=
by {
  sorry
}

end arrange_exponents_l508_508447


namespace disjoint_triangles_impossible_l508_508835

-- Defining the points and their relationships
variables (A B C M N P : Type) [Point A] [Point B] [Point C]
[Point M] [Point N] [Point P]
(midpoint_AB : isMidpoint A B M)
(midpoint_BC : isMidpoint B C N)
(midpoint_CA : isMidpoint C A P)

theorem disjoint_triangles_impossible :
  ∀ (A B C M N P : Type) [Point A] [Point B] [Point C] [Point M] [Point N] [Point P],
  (isMidpoint A B M) → (isMidpoint B C N) → (isMidpoint C A P) →
  ¬ (Exists (triangle₁ triangle₂ : Triangle),
    disjoint triangle₁ triangle₂ ∧
    triangle_vertices triangle₁ = {A, B, C} ∪ {M, N, P} ∧ triangle_vertices triangle₂ = {A, B, C} ∪ {M, N, P}) :=
sorry

end disjoint_triangles_impossible_l508_508835


namespace inequality_1_inequality_2a_inequality_2b_l508_508611

theorem inequality_1 (x : ℝ) : 2^(3*x - 1) < 2 → x < 2/3 := 
sorry

theorem inequality_2a (x a : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1) (h₃ : 1 < a) :
  a^(3*x^2 + 3*x - 1) < a^(3*x^2 + 3) → x < 4/3 := 
sorry

theorem inequality_2b (x a : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1) (h₃ : a < 1) :
  a^(3*x^2 + 3*x - 1) < a^(3*x^2 + 3) → x > 4/3 := 
sorry

end inequality_1_inequality_2a_inequality_2b_l508_508611


namespace triangle_area_equivalence_l508_508526

theorem triangle_area_equivalence :
  ∃ (BC : ℝ), 
  let A := 1/√2 * BC * sqrt2 + 1/2 * BC * sin (40 * π/180) in
  (∃ (E : ℝ × ℝ), ∃ (D : ℝ × ℝ), 
    let is_midpoint := ∃ (B C : ℝ × ℝ), E = (B + C) / 2 in
    let is_isosceles := ∃ (A C : ℝ × ℝ), D = (A + C) / 2 in
    let given_conditions := 
        AC = 2 ∧  ∠BAC = 45 ∧ ∠ABC = 85 ∧ ∠ACB = 50 ∧ ∠DEC = 70 in
    is_midpoint ∧ is_isosceles ∧ given_conditions) -> 
  A = (√2 / 2) + ((1 / 2) * sin(40 * π / 180)) := 
by
  sorry

end triangle_area_equivalence_l508_508526


namespace suzy_twice_mary_l508_508939

def suzy_current_age : ℕ := 20
def mary_current_age : ℕ := 8

theorem suzy_twice_mary (x : ℕ) : suzy_current_age + x = 2 * (mary_current_age + x) ↔ x = 4 := by
  sorry

end suzy_twice_mary_l508_508939


namespace distance_from_focus_l508_508816

theorem distance_from_focus (y : ℝ) (hyp : y ^ 2 = 12) : 
  real.sqrt ((3 - 1) ^ 2 + y ^ 2) = 4 := by
  sorry

end distance_from_focus_l508_508816


namespace range_of_a_l508_508216

open Real

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 
  if x % 5 = 2 then sorry else if x % 5 = 3 then (a^2 + a + 3) / (a - 3) else sorry

theorem range_of_a (a : ℝ) :
  (f 2 a) > 1 ∧ f x a = f (x + 5) a ∧ (∀ x, f (-x) a = -f x a) →
  a ∈ (-∞, -2) ∪ (0, 3) :=
by sorry

end range_of_a_l508_508216


namespace cos_four_theta_l508_508887

theorem cos_four_theta (θ : ℝ) (h : complex.exp (complex.I * θ) = (3 + complex.I * real.sqrt 8) / 4) : 
  real.cos (4 * θ) = -287 / 256 := by sorry

end cos_four_theta_l508_508887


namespace triangle_intersection_ratio_l508_508174

noncomputable def condition1 (CD DB : ℝ) : Prop := CD / DB = 2
noncomputable def condition2 (AF FB : ℝ) : Prop := AF / FB = 5/3
noncomputable def find_s (CQ QF : ℝ) : Prop := CQ / QF = 1.8

theorem triangle_intersection_ratio 
  (A B C D F Q : Point)
  (h1 : condition1 (CD D B))
  (h2 : condition2 (AF A F))
  (hQ : intersection_point Q CF AD) :
  find_s (CQ C Q) (QF Q F) :=
  sorry

end triangle_intersection_ratio_l508_508174


namespace same_terminal_side_angle_l508_508617

theorem same_terminal_side_angle (k : ℤ) : 
  ∃ (θ : ℤ), θ = k * 360 + 257 ∧ (θ % 360 = (-463) % 360) :=
by
  sorry

end same_terminal_side_angle_l508_508617


namespace calc_x_squared_y_squared_l508_508883

theorem calc_x_squared_y_squared (x y : ℝ) (h1 : (x + y)^2 = 4) (h2 : x * y = -9) : x^2 + y^2 = 22 := by
  sorry

end calc_x_squared_y_squared_l508_508883


namespace addilynn_eggs_initial_l508_508027

theorem addilynn_eggs_initial (E : ℕ) (H1 : ∃ (E : ℕ), (E / 2) - 15 = 21) : E = 72 :=
by
  sorry

end addilynn_eggs_initial_l508_508027


namespace division_quotient_example_l508_508180

theorem division_quotient_example :
  ∃ q : ℕ,
    let dividend := 760
    let divisor := 36
    let remainder := 4
    dividend = divisor * q + remainder ∧ q = 21 :=
by
  sorry

end division_quotient_example_l508_508180


namespace max_F_of_eternal_number_l508_508905

-- Definitions of digits and the number M
def is_eternal_number (a b c d : ℕ) : Prop :=
  b + c + d = 12 ∧ a = b - d

-- Definition of the function F(M)
def F (a b c d : ℕ) : ℤ :=
  100 * a - 100 * b + c - d

-- The mathematical statement to be proved
theorem max_F_of_eternal_number (a b c d : ℕ) (h1 : is_eternal_number a b c d)
    (h2 : ∃ k : ℤ, F a b c d = 9 * k) : 
    F a b c d ≤ 9 :=
begin
  sorry
end

end max_F_of_eternal_number_l508_508905


namespace min_average_cost_min_production_A_l508_508693

open Real

def cost_A (x : ℝ) : ℝ := 4 * x ^ 2
def cost_B (x : ℝ) : ℝ := 8 * (x + 2)
def average_cost (x : ℝ) : ℝ := (cost_A x + cost_B x) / (2 * x + 2)

theorem min_average_cost :
  (∀ x : ℝ, x ∈ set.Ioi 0 → average_cost x ≥ 4 * sqrt 3) ∧ (average_cost (sqrt 3 - 1) = 4 * sqrt 3) := 
sorry

theorem min_production_A (x : ℝ) :
  (x ∈ set.Icc 0 (1 / 2) ∨ x ∈ set.Icc 2 8) →
  (average_cost x ≤ average_cost y ∀ y : ℝ, y ∈ set.Icc 0 (1 / 2) ∨ y ∈ set.Icc 2 8 → x = 1 / 2) := 
sorry

end min_average_cost_min_production_A_l508_508693


namespace exists_wonderful_with_prime_factors_l508_508971

theorem exists_wonderful_with_prime_factors :
  ∃ (n : ℕ), (n > 1) ∧ (∀ (factors : ℕ → ℕ), (∀ i, factors i is_prime) ∧ (finset.card (finset.filter is_prime (finset.range (n+1))) ≥ 10^2002) ∧ (n % (finset.sum (finset.filter is_prime (finset.range (n+1)))) = 0)) := sorry

end exists_wonderful_with_prime_factors_l508_508971


namespace find_m_l508_508490

open Real

noncomputable def curve_equation (x y : ℝ) : Prop :=
  (x - 1)^2 + y^2 = 1

noncomputable def line_equation (m t x y : ℝ) : Prop :=
  x = (sqrt 3 / 2) * t + m ∧ y = (1 / 2) * t

noncomputable def dist (x1 y1 x2 y2 : ℝ) : ℝ :=
  sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem find_m (m : ℝ) (h_nonneg : 0 ≤ m) :
  (∀ (t1 t2 : ℝ), (∀ x y, line_equation m t1 x y → curve_equation x y) → 
                   (∀ x y, line_equation m t2 x y → curve_equation x y) →
                   (dist m 0 x1 y1) * (dist m 0 x2 y2) = 1) →
  m = 1 ∨ m = 1 + sqrt 2 :=
sorry

end find_m_l508_508490


namespace trip_time_is_14_l508_508687

-- Define the conditions
def avg_speed1 := 40  -- miles per hour
def time1 := 4  -- hours
def avg_speed2 := 65  -- miles per hour
def time2 := 3  -- hours
def avg_speed3 := 54  -- miles per hour
def time3 := 2  -- hours
def avg_speed4 := 70  -- miles per hour
def total_avg_speed := 58  -- miles per hour

-- Define the total time for a car trip
noncomputable def total_trip_time (x : ℕ) : ℕ :=
  time1 + time2 + time3 + x

-- Define the total distance
noncomputable def total_distance (x : ℕ) : ℕ :=
  avg_speed1 * time1 + avg_speed2 * time2 + avg_speed3 * time3 + avg_speed4 * x

-- Statement of the problem to prove
theorem trip_time_is_14 : ∃ x, total_avg_speed = total_distance x / total_trip_time x ∧ total_trip_time x = 14 := by
  sorry


end trip_time_is_14_l508_508687


namespace range_of_m_l508_508286

noncomputable def quadratic_fn {α : Type*} [linear_ordered_field α] (f : α → α) : Prop :=
quadratic f ∧ (∀ x, f (4 + x) = f (-x)) ∧ f 2 = 1 ∧ f 0 = 3 ∧ f 4 = 3

theorem range_of_m {α : Type*} [linear_ordered_field α] (f : α → α) :
  quadratic_fn f → ∀ m, (∀ x, 0 ≤ x ∧ x ≤ m → (1 ≤ f x ∧ f x ≤ 3)) ↔ (2 ≤ m ∧ m ≤ 4) :=
by
  intro h_quadratic_fn
  sorry

end range_of_m_l508_508286


namespace constant_term_of_binomial_expansion_l508_508082

-- Define the binomial coefficient function binom
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Define the term in the binomial expansion of (x^2 - 1/x)^6
def term (r : ℕ) := (-1)^r * binom 6 r

-- Define the condition which finds the constant term (exponent 0)
def is_constant_term (r : ℕ) : Prop :=
  12 - 3 * r = 0

theorem constant_term_of_binomial_expansion : ∃ r : ℕ, is_constant_term r ∧ term r = 15 := by
  -- r = 4 satisfies the is_constant_term condition
  use 4
  split
  -- Prove the condition 12 - 3r = 0
  {
    sorry -- Condition checking goes here
  }
  -- Prove the term's value
  {
    sorry -- Proof of term value goes here
  }

end constant_term_of_binomial_expansion_l508_508082


namespace common_rational_root_l508_508628

-- Definitions for the given conditions
def polynomial1 (a b c : ℤ) (x : ℚ) := 50 * x^4 + a * x^3 + b * x^2 + c * x + 16 = 0
def polynomial2 (d e f g : ℤ) (x : ℚ) := 16 * x^5 + d * x^4 + e * x^3 + f * x^2 + g * x + 50 = 0

-- The proof problem statement: Given the conditions, proving that -1/2 is a common rational root
theorem common_rational_root (a b c d e f g : ℤ) (k : ℚ) 
  (h1 : polynomial1 a b c k)
  (h2 : polynomial2 d e f g k) 
  (h3 : ∃ m n : ℤ, k = -((m : ℚ) / n) ∧ Int.gcd m n = 1) :
  k = -1/2 :=
sorry

end common_rational_root_l508_508628


namespace perimeter_of_PGHI_l508_508910

noncomputable def P := (0, 0)
noncomputable def Q := (17, 0)
noncomputable def R := (8.5, 15)

-- Given conditions
def triangle_PQR : Prop := dist P Q = 17 ∧ dist P R = 17 ∧ dist Q R = 15

def on_PQ (G : (ℝ × ℝ)) : Prop := ∃ λ, 0 ≤ λ ∧ λ ≤ 1 ∧ G = λ • P + (1 - λ) • Q
def on_QR (H : (ℝ × ℝ)) : Prop := ∃ μ, 0 ≤ μ ∧ μ ≤ 1 ∧ H = μ • Q + (1 - μ) • R
def on_PR (I : (ℝ × ℝ)) : Prop := ∃ ν, 0 ≤ ν ∧ ν ≤ 1 ∧ I = ν • P + (1 - ν) • R

def GH_parallel_PR (G H I : (ℝ × ℝ)) : Prop := H.2 = G.2 ∧ H.2 = I.2 -- assuming a coordinate system where PR is parallel to the x-axis
def HI_parallel_PQ (G H I : (ℝ × ℝ)) : Prop := H.1 = I.1 ∧ I.1 = G.1 + 17 / 2 -- assuming similar coordinates

-- Proving the perimeter of PGHI equals 34
theorem perimeter_of_PGHI
  (G H I : (ℝ × ℝ))
  (hPQR : triangle_PQR)
  (hG : on_PQ G)
  (hH : on_QR H)
  (hI : on_PR I)
  (hGH_PR : GH_parallel_PR G H I)
  (hHI_PQ : HI_parallel_PQ G H I) :
  dist P G + dist G H + dist H I + dist I P = 34 := sorry

end perimeter_of_PGHI_l508_508910


namespace ratio_of_terms_l508_508838

noncomputable def geometric_sum (a₁ q : ℝ) (n : ℕ) : ℝ := a₁ * (1 - q^n) / (1 - q)

theorem ratio_of_terms
  (a : ℕ → ℝ) (b : ℕ → ℝ)
  (S T : ℕ → ℝ)
  (h₀ : ∀ n : ℕ, S n = geometric_sum (a 1) (a 2) n)
  (h₁ : ∀ n : ℕ, T n = geometric_sum (b 1) (b 2) n)
  (h₂ : ∀ n : ℕ, n > 0 → S n / T n = (3 ^ n + 1) / 4) :
  a 3 / b 4 = 3 := 
sorry

end ratio_of_terms_l508_508838


namespace product_floor_ceil_sequence_l508_508411

noncomputable def floor (x : ℝ) : ℤ := Int.floor x
noncomputable def ceil (x : ℝ) : ℤ := Int.ceil x

theorem product_floor_ceil_sequence :
    (floor (-6 - 0.5) * ceil (6 + 0.5)) *
    (floor (-5 - 0.5) * ceil (5 + 0.5)) *
    (floor (-4 - 0.5) * ceil (4 + 0.5)) *
    (floor (-3 - 0.5) * ceil (3 + 0.5)) *
    (floor (-2 - 0.5) * ceil (2 + 0.5)) *
    (floor (-1 - 0.5) * ceil (1 + 0.5)) *
    (floor (-0.5) * ceil (0.5)) = -25401600 :=
by
  sorry

end product_floor_ceil_sequence_l508_508411


namespace sequence_solution_l508_508671

def sequence_conditions (s : List ℕ) : Prop :=
  (∀ n, 1 ≤ n ∧ n ≤ 9 → (count s n = 3)) ∧
  (∀ n, 1 ≤ n ∧ n ≤ 9 →
    let idx := List.findIdxs (λ x => x = n) s in
    ((idx.length = 3) ∧ -- There must be three occurrences
     (idx !! 1 = some (idx.head + n + 1)) ∧ -- Check distance between 1st and 2nd occurrence
     (idx !! 2 = some (idx !! 1 + n + 1))))

def partial_sequence : List ℕ :=
  [_, _, _, _, _, _, _, _, _, 5, 7, 4, 6, 9, 2, 5, 8, _, _, _, _, _, _, _, _, _, _]

def complete_sequence : List ℕ :=
  [3, 4, 7, 9, 3, 6, 4, 8, 3, 5, 7, 4, 6, 9, 2, 5, 8, 2, 7, 6, 2, 5, 1, 9, 1, 8, 1]

theorem sequence_solution :
  sequence_conditions complete_sequence :=
  sorry

end sequence_solution_l508_508671


namespace complement_of_A_l508_508117

open Set

variable (U : Set ℝ := univ)
noncomputable def A : Set ℝ := {x | x^2 + 3 * x ≥ 0} ∪ {x | 2^x > 1}

theorem complement_of_A : U \ A = {x | -3 < x ∧ x < 0} := 
by 
  sorry

end complement_of_A_l508_508117


namespace dihedral_angles_pyramid_l508_508083

noncomputable def dihedral_angles (a b : ℝ) : ℝ × ℝ :=
  let alpha := Real.arccos ((a * Real.sqrt 3) / Real.sqrt (4 * b ^ 2 - a ^ 2))
  let gamma := 2 * Real.arctan (b / Real.sqrt (4 * b ^ 2 - a ^ 2))
  (alpha, gamma)

theorem dihedral_angles_pyramid (a b alpha gamma : ℝ) (h1 : a > 0) (h2 : b > 0) :
  dihedral_angles a b = (alpha, gamma) :=
sorry

end dihedral_angles_pyramid_l508_508083


namespace no_perfect_square_in_sequence_l508_508729

-- Define the sequence using its recurrence relation.
def sequence (n : ℕ) : ℤ :=
  match n with
  | 0     => 2    -- x₁ = 2
  | 1     => 7    -- x₂ = 7
  | n + 2 => 4 * sequence (n + 1) - sequence n

-- Define a predicate to check if an integer is a perfect square.
def is_perfect_square (n : ℤ) : Prop :=
  ∃ k : ℤ, k * k = n

-- Prove that there does not exist a perfect square in the sequence.
theorem no_perfect_square_in_sequence : ∀ n : ℕ, ¬is_perfect_square (sequence n) := 
by
  -- This is where the proof would go
  sorry

end no_perfect_square_in_sequence_l508_508729


namespace coeff_x_squared_l508_508270

theorem coeff_x_squared (x y : ℝ) : 
    (let term r := (Nat.choose 8 r) * (x / y)^(8 - r) * (-y / (real.sqrt x))^r in
     ∃ r, (8 - 3 * r / 2 = 2) ∧ (Nat.choose 8 r = 70)) :=
sorry

end coeff_x_squared_l508_508270


namespace range_of_a_l508_508856

noncomputable def interval_contains_two_integers (a : ℝ) :=
  ∃ x₁ x₂ : ℝ, x₁ < x₂ ∧
    (x₁ < 2 ∧ 4 < x₂ ∧ ∀ k, x₁ < k → k < x₂ → k = 2 ∨ k = 4) ∧
    (x^2 - 2*a*x + 15 - 2*a).roots = [x₁, x₂]

theorem range_of_a :
  {a : ℝ | interval_contains_two_integers a} = {a : ℝ | (31/10 : ℝ) < a ∧ a ≤ 19/6} :=
begin
  sorry
end

end range_of_a_l508_508856


namespace scientific_notation_flu_virus_diameter_l508_508950

theorem scientific_notation_flu_virus_diameter :
  0.000000823 = 8.23 * 10^(-7) :=
sorry

end scientific_notation_flu_virus_diameter_l508_508950


namespace magician_finds_coins_l508_508360

/-!
# Problem Statement
Given 13 close boxes in a circular arrangement, two of which contain coins, and an assistant who opens 
one box that does not contain a coin, we aim to prove the magician can always identify the two boxes 
containing the coins by predefined rules and template rotation.

The question is formulated as a theorem:
The magician can always identify the two boxes with the coins.
-/

def box := fin 13

def contains_coin (b : box) : Prop := sorry
def predefined_template (b : box) (n : ℕ) : box := sorry
def assistant_opens (b : box) : Prop := ¬ contains_coin b

theorem magician_finds_coins (b1 b2 : box) (a : box) :
 (contains_coin b1) ∧ (contains_coin b2) ∧ ¬(contains_coin a) 
 ∧ (a ≠ b1) ∧ (a ≠ b2) → (∃ (k : ℕ), 
    (predefined_template a k = b1 ∨ predefined_template a k = b2) ∧ 
    (predefined_template a (k+1) = b1 ∨ predefined_template a (k+1) = b2) ∧ 
    (predefined_template a (k+2) = b1 ∨ predefined_template a (k+2) = b2) ∧ 
    (predefined_template a (k+3) = b1 ∨ predefined_template a (k+3) = b2)
) :=
sorry

end magician_finds_coins_l508_508360


namespace necessary_and_sufficient_condition_l508_508589

def U (a : ℕ) : Set ℕ := { x | x ≤ a ∧ x > 0 }
def P := {1, 2, 3}
def Q := {4, 5, 6}
def complement_U (U : Set ℕ) (P : Set ℕ) : Set ℕ := U \ P

theorem necessary_and_sufficient_condition (a : ℕ) :
  (a ≥ 6 ∧ a < 7) ↔ (complement_U (U a) P = Q) := sorry

end necessary_and_sufficient_condition_l508_508589


namespace ellipse_locus_l508_508498

noncomputable def F1 : ℝ × ℝ := (-1, 0)
noncomputable def F2 : ℝ × ℝ := (1, 0)

theorem ellipse_locus :
  (∃ P : ℝ × ℝ, (∥P - F1∥ + ∥P - F2∥) / 2 = ∥F1 - F2∥) →
  (∀ P : ℝ × ℝ, ∥P - F1∥ + ∥P - F2∥ = 4 → (P.1^2) / 16 + (P.2^2) / 9 = 1) :=
by
  sorry

end ellipse_locus_l508_508498


namespace x_mul_one_minus_f_eq_1024_l508_508578

noncomputable def alpha := 2 + Real.sqrt 2
noncomputable def beta := 2 - Real.sqrt 2

theorem x_mul_one_minus_f_eq_1024 :
  let x := alpha ^ 10
  let n := Int.floor x
  let f := x - n
  x * (1 - f) = 1024 := by
  sorry

end x_mul_one_minus_f_eq_1024_l508_508578


namespace solve_x_in_equation_l508_508609

theorem solve_x_in_equation : ∃ (x : ℤ), 24 - 4 * 2 = 3 + x ∧ x = 13 :=
by
  use 13
  sorry

end solve_x_in_equation_l508_508609


namespace fraction_of_remaining_supplies_used_l508_508373

theorem fraction_of_remaining_supplies_used 
  (initial_food : ℕ)
  (food_used_first_day_fraction : ℚ)
  (food_remaining_after_three_days : ℕ) 
  (food_used_second_period_fraction : ℚ) :
  initial_food = 400 →
  food_used_first_day_fraction = 2 / 5 →
  food_remaining_after_three_days = 96 →
  (initial_food - initial_food * food_used_first_day_fraction) * (1 - food_used_second_period_fraction) = food_remaining_after_three_days →
  food_used_second_period_fraction = 3 / 5 :=
by
  intros h1 h2 h3 h4
  sorry

end fraction_of_remaining_supplies_used_l508_508373


namespace intersection_A_B_l508_508865

open Set

def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {x | ∃ a ∈ A, x = 2 * a}

theorem intersection_A_B : A ∩ B = {0, 2} :=
by
  sorry

end intersection_A_B_l508_508865


namespace darry_full_ladder_steps_l508_508767

theorem darry_full_ladder_steps
  (climbs_full_ladder : ℕ)
  (climbs_smaller_ladder : ℕ)
  (full_ladder_steps : ℕ → ℕ)
  (smaller_ladder_steps : ℕ)
  (total_steps : ℕ) :
  climbs_full_ladder = 10 →
  climbs_smaller_ladder = 7 →
  smaller_ladder_steps = 6 →
  total_steps = 152 →
  ∀ x, full_ladder_steps x = x →
  10 * x + 7 * 6 = 152 →
  x = 11 :=
begin
  intros h1 h2 h3 h4 h_steps hx,
  -- proof will go here
  sorry
end

end darry_full_ladder_steps_l508_508767


namespace total_cost_correct_l508_508698

noncomputable def total_cost_of_laying_floor : ℝ :=
  let area1 := 5.5 * 3.75 in
  let cost1 := area1 * 400 in
  let area2 := (4 * 2.5) + (2 * 1.5) in
  let cost2 := area2 * 350 in
  let area3 := (3.5 * 2) / 2 in
  let cost3 := area3 * 450 in
  cost1 + cost2 + cost3

theorem total_cost_correct : 
  total_cost_of_laying_floor = 14375 :=
  sorry -- proof not required

end total_cost_correct_l508_508698


namespace count_valid_c_values_for_equation_l508_508804

theorem count_valid_c_values_for_equation :
  ∃ (c_set : Finset ℕ), (∀ c ∈ c_set, c ≤ 2000 ∧
    ∃ x : ℝ, 5 * (Real.floor x) + 3 * (Real.ceil x) = c) ∧
    c_set.card = 500 :=
by
  sorry

end count_valid_c_values_for_equation_l508_508804


namespace complex_real_imag_parts_l508_508520

-- Definition of the complex number z
def z : ℂ := 2 - 3 * Complex.i

-- Statement of the proof problem
theorem complex_real_imag_parts : (z.re = 2) ∧ (z.im = -3) :=
by
  -- Proof is omitted
  sorry

end complex_real_imag_parts_l508_508520


namespace total_duration_of_running_l508_508351

-- Definition of conditions
def constant_speed_1 : ℝ := 18
def constant_time_1 : ℝ := 3
def next_distance : ℝ := 70
def average_speed_2 : ℝ := 14

-- Proof statement
theorem total_duration_of_running : 
    let distance_1 := constant_speed_1 * constant_time_1
    let time_2 := next_distance / average_speed_2
    distance_1 = 54 ∧ time_2 = 5 → (constant_time_1 + time_2 = 8) :=
sorry

end total_duration_of_running_l508_508351


namespace arithmetic_sequence_sum_l508_508820

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (d : ℕ) (h1 : a 1 + a 13 = 10) 
  (h2 : ∀ n m : ℕ, a (n + 1) = a n + d) : a 3 + a 5 + a 7 + a 9 + a 11 = 25 :=
  sorry

end arithmetic_sequence_sum_l508_508820


namespace bob_weekly_profit_l508_508752

theorem bob_weekly_profit :
  let daily_muffins := 12
  let cost_per_muffin := 0.75
  let sell_price_per_muffin := 1.5
  let days_per_week := 7
  let daily_cost := daily_muffins * cost_per_muffin
  let daily_revenue := daily_muffins * sell_price_per_muffin
  let daily_profit := daily_revenue - daily_cost
  let weekly_profit := daily_profit * days_per_week
  in weekly_profit = 63 :=
by
  sorry

end bob_weekly_profit_l508_508752


namespace find_monic_quadratic_polynomial_l508_508423

noncomputable def monic_quadratic_polynomial_with_root (x : ℂ) : Prop :=
  ∃ (polynomial : ℂ → ℂ), polynomial = (λ x, x^2 - 6*x + 16) ∧ polynomial (3 - complex.i * sqrt 7) = 0

theorem find_monic_quadratic_polynomial :
  ∃ polynomial : (ℂ → ℂ), polynomial = (λ x, x^2 - 6*x + 16) ∧ polynomial (3 - complex.i * sqrt 7) = 0 :=
begin
  use (λ x, x^2 - 6*x + 16),
  split,
  { refl },
  { simp [complex.i] }
end

end find_monic_quadratic_polynomial_l508_508423


namespace coefficient_of_x3y2_in_expansion_l508_508399

theorem coefficient_of_x3y2_in_expansion : 
  (coeff ((x - 2 * y) ^ 5) (monomial 3 2)) = 40 := 
by 
  sorry

end coefficient_of_x3y2_in_expansion_l508_508399


namespace farmer_transaction_percent_gain_l508_508357

theorem farmer_transaction_percent_gain (x : ℝ) (h₁ : x > 0) : 
    let total_cost := 800 * x + 100 * x in
    let revenue_750_sheep := total_cost in
    let price_per_sheep := revenue_750_sheep / 750 in
    let revenue_50_sheep := 50 * price_per_sheep in
    let total_revenue := revenue_750_sheep + revenue_50_sheep in
    let profit := total_revenue - total_cost in
    let percent_gain := (profit / total_cost) * 100 in
    percent_gain = 6.67 :=
by
  sorry

end farmer_transaction_percent_gain_l508_508357


namespace digit_2023_of_7_over_12_l508_508789

theorem digit_2023_of_7_over_12 : (decimal_expansion_digit 2023 (7 / 12)) = 3 := 
sorry

end digit_2023_of_7_over_12_l508_508789


namespace x_plus_one_div_x_equals_five_l508_508847

theorem x_plus_one_div_x_equals_five (x : ℝ) (hx : x ≠ 0) (w : ℝ) 
    (hw : w = x^2 + (1/x)^2) : (x + 1/x = 5) :=
begin
  have key : (x + 1/x)^2 - 2 = w,
  { rw [hw, ← sq x, ← sq (1/x), ← add_sq_eq],
    norm_cast,
    ring },
  have zero_lt_w : 0 < w,
    from (sq_nonneg x).add (sq_nonneg (1/x)),
  field_simp at hw,
  norm_cast at hw,
  have zero_lt_w : 0 < x + 1/x,
    from add_pos (zero_lt_sq (1/x)) zero_lt_w,
  exact add_self_inj.1 (hw.trans (zero_lt_w.symm.ne)),
end

end x_plus_one_div_x_equals_five_l508_508847


namespace cost_of_square_fence_l508_508661

noncomputable def cost_of_fence (area : ℝ) (price_per_foot : ℝ) : ℝ :=
  let side_length := Real.sqrt area
  let perimeter := 4 * side_length
  perimeter * price_per_foot

theorem cost_of_square_fence (h_area : Real.sqrt 289 = 17) (price_per_foot : ℝ) : cost_of_fence 289 price_per_foot = 3672 :=
by
  have h : 4 * 17 = 68 := by norm_num
  rw [cost_of_fence, h_area]
  simp only [Real.sqrt_sixty_four, mul_comm, mul_assoc, h]
  norm_num
  sorry

end cost_of_square_fence_l508_508661


namespace socks_pairing_l508_508936

open Finset

def sockColors : Type := ℕ × ℕ × ℕ × ℕ  -- (white, green, brown, blue)

def socks : sockColors := (6, 4, 4, 2)

def choose_2 (n : ℕ) : ℕ := n * (n - 1) / 2

def sameColorPairs (s : sockColors) : ℕ :=
  choose_2 s.1 + choose_2 s.2 + choose_2 s.3 + choose_2 s.4

theorem socks_pairing : sameColorPairs socks = 28 :=
by
  sorry

end socks_pairing_l508_508936


namespace length_BC_fraction_AD_l508_508602

/-- Points B and C lie on line segment AD such that
    - The length of AB is 3 times the length of BD
    - The length of AC is 5 times the length of CD,
    and BC overlaps CD. We prove that the length of BC is 
    1/12 of the length of AD. -/
theorem length_BC_fraction_AD 
  (A B C D : Type)
  [metric_space A] 
  [metric_space B] 
  [metric_space C] 
  [metric_space D]
  (AB BD AC CD AD BC : ℝ) 
  (h1 : AB = 3 * BD)
  (h2 : AC = 5 * CD)
  (h3 : BC = AC - AB)
  (h4 : AD = AB + BD)
  (h5 : AD = AC + CD) :
  BC = (1 / 12) * AD := 
sorry

end length_BC_fraction_AD_l508_508602


namespace min_games_for_sharks_winning_percentage_l508_508265

theorem min_games_for_sharks_winning_percentage
  (initial_wins initial_games : ℕ)
  (sharks_wins_after : ℕ)
  (N : ℕ)
  (total_games : ℕ)
  (win_condition : (2 : ℚ) / (3 + N : ℚ) ≥ (9 / 10 : ℚ))
  (initial_wins = 2)
  (initial_games = 3)
  (sharks_wins_after = 2)
  (total_games = 3 + N) : 
  N = 7 := by
    sorry

end min_games_for_sharks_winning_percentage_l508_508265


namespace simplify_expr_l508_508257

theorem simplify_expr : 3 * (4 - 2 * Complex.I) - 2 * Complex.I * (3 - 2 * Complex.I) = 8 - 12 * Complex.I :=
by
  sorry

end simplify_expr_l508_508257


namespace sum_of_series_l508_508756

theorem sum_of_series :
  (∑ k in Finset.range 2007, 1 / (k + 1) * 1 / (k + 2)) = 2007 / 2008 := by
  sorry

end sum_of_series_l508_508756


namespace find_n_l508_508948

theorem find_n :
  ∃ (n : ℕ), 
  ∃ (a : fin (n+2) → ℤ),  
  a 0 = -9 ∧ a (fin.last (n+1)) = 3 ∧ 
  (∀ (i : fin (n+1)), a (i.succ) - a i = a 1 - a 0) ∧ 
  (∑ i, a i) = -21 ∧ 
  n = 5 := 
sorry

end find_n_l508_508948


namespace complement_intersection_l508_508868

def M : Set ℝ := { x | x ≥ 1 }
def N : Set ℝ := { x | x < 2 }
def CR (S : Set ℝ) : Set ℝ := { x | x ∉ S }

theorem complement_intersection :
  CR (M ∩ N) = { x | x < 1 } ∪ { x | x ≥ 2 } := by
  sorry

end complement_intersection_l508_508868


namespace discrete_random_variables_l508_508382

/-- Definition of a discrete random variable:
A variable that can take on only a countable number of distinct values. --/
def is_discrete (X : Type) : Prop :=
  ∃ s : set X, countable s ∧ ∀ x : X, x ∈ s

variables (X1 X2 X3 : Type)

/-- Defining the scenarios as variables --/
def passengers (X : Type) := is_discrete X
def daily_hits (X : Type) := is_discrete X
def water_level (X : Type) := ¬ is_discrete X

theorem discrete_random_variables :
  passengers X1 ∧ daily_hits X2 ∧ water_level X3 → X1 ∧ X2 :=
begin
  sorry
end

end discrete_random_variables_l508_508382


namespace train_distance_l508_508380

theorem train_distance (D : ℝ) (h_speed1 : 160 > 0) (h_speed2 : 120 > 0)
  (h_trip_back_longer : D / 120 = D / 160 + 1) : D = 480 :=
begin
  sorry
end

end train_distance_l508_508380


namespace total_students_count_l508_508643

-- Define the conditions
def num_rows : ℕ := 8
def students_per_row : ℕ := 6
def students_last_row : ℕ := 5
def rows_with_six_students : ℕ := 7

-- Define the total students
def total_students : ℕ :=
  (rows_with_six_students * students_per_row) + students_last_row

-- The theorem to prove
theorem total_students_count : total_students = 47 := by
  sorry

end total_students_count_l508_508643


namespace find_g_l508_508124

noncomputable def f (x : ℝ) : ℝ := sqrt 3 * sin (2 * x) + 2 * cos x ^ 2

noncomputable def f₁ (x : ℝ) : ℝ := f (x - π / 12)

noncomputable def g (x : ℝ) : ℝ := 2 * sin x + 1

theorem find_g :
  g x = 2 * sin x + 1 :=
sorry

end find_g_l508_508124


namespace eval_f_at_neg_3_l508_508123

def f (x : ℝ) : ℝ := if x >= 0 then x * (x + 1) else x * (1 - x)

theorem eval_f_at_neg_3 : f (-3) = -12 := by
  sorry

end eval_f_at_neg_3_l508_508123


namespace minimum_value_expression_l508_508982

theorem minimum_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  ∃ k, k = 729 ∧ ∀ x y z, 0 < x → 0 < y → 0 < z → k ≤ (x^2 + 4*x + 4) * (y^2 + 4*y + 4) * (z^2 + 4*z + 4) / (x * y * z) :=
by 
  use 729
  sorry

end minimum_value_expression_l508_508982


namespace line_circle_separation_l508_508098

theorem line_circle_separation (a b : ℝ) (h : a^2 + b^2 < 1) :
    let d := 1 / (Real.sqrt (a^2 + b^2))
    d > 1 := by
    sorry

end line_circle_separation_l508_508098


namespace find_lambda_l508_508464

open_locale real

variables {A B C D : Type} [AddCommGroup A] [Module ℝ A] (a b c d : A)
variables (h1 : ∃ m : ℝ, d = -m • (c - a) + (m - 1) • (b - a))

theorem find_lambda (λ : ℝ) (h2 : d = 1/3 • (c - a) + λ • (b - c)) :
  λ = -4/3 :=
by 
  -- sorry placeholder for the proof steps
  sorry

end find_lambda_l508_508464


namespace perp_PD_EF_l508_508138

-- Definitions for the angles based on the conditions given in the problem
variables {α β γ : ℝ}

-- Main theorem statement
theorem perp_PD_EF
  (h1 : ∠BAC = α)
  (h2 : ∠ABC = β)
  (h3 : ∠ACB = γ)
  (h4 : ∠PBC = 90 - β / 2)
  (h5 : ∠EAC = 90 - β / 2)
  (h6 : ∠ECA = 90 - β / 2)
  (h7 : ∠PCB = 90 - γ / 2)
  (h8 : ∠FAB = 90 - γ / 2)
  (h9 : ∠FBA = 90 - γ / 2)
  (h10 : ∠DBC = α / 2)
  (h11 : ∠DCB = α / 2) :
  PD ⊥ EF :=
  sorry

end perp_PD_EF_l508_508138


namespace traffic_lights_states_l508_508537

theorem traffic_lights_states (n k : ℕ) : 
  (k ≤ n) → 
  (∃ (ways : ℕ), ways = 3^k * 2^(n - k)) :=
by
  sorry

end traffic_lights_states_l508_508537


namespace bread_consumption_snacks_per_day_l508_508292

theorem bread_consumption_snacks_per_day (members : ℕ) (breakfast_slices_per_member : ℕ) (slices_per_loaf : ℕ) (loaves : ℕ) (days : ℕ) (total_slices_breakfast : ℕ) (total_slices_all : ℕ) (snack_slices_per_member_per_day : ℕ) :
  members = 4 →
  breakfast_slices_per_member = 3 →
  slices_per_loaf = 12 →
  loaves = 5 →
  days = 3 →
  total_slices_breakfast = members * breakfast_slices_per_member * days →
  total_slices_all = slices_per_loaf * loaves →
  snack_slices_per_member_per_day = ((total_slices_all - total_slices_breakfast) / members / days) →
  snack_slices_per_member_per_day = 2 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  -- We can insert the proof outline here based on the calculations from the solution steps
  sorry

end bread_consumption_snacks_per_day_l508_508292


namespace bob_time_improvement_l508_508037

def time_improvement_percent (bob_time sister_time improvement_time : ℕ) : ℕ :=
  ((improvement_time * 100) / bob_time)

theorem bob_time_improvement : 
  ∀ (bob_time sister_time : ℕ), bob_time = 640 → sister_time = 608 → 
  time_improvement_percent bob_time sister_time (bob_time - sister_time) = 5 :=
by
  intros bob_time sister_time h_bob h_sister
  rw [h_bob, h_sister]
  sorry

end bob_time_improvement_l508_508037


namespace problem_conditions_l508_508483

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x - a * x^2

theorem problem_conditions (a x1 x2 : ℝ) (h1 : x1 < x2) 
                           (h2 : ∃ x, f' a x = 0 ∧ x = x1 ∨ x = x2) :
  0 < a ∧ a < 1/2 ∧ (∃ a, (x1 + x2 = x2 / x1)) := by 
  sorry

end problem_conditions_l508_508483


namespace zero_on_interval_l508_508226

-- Define the conditions as a Lean structure
structure ContinuousFunction (f : ℝ → ℝ) : Prop :=
  (continuous : Continuous f)
  (functional_eq : ∀ x, f (2 * x^2 - 1) = 2 * x * f x)

-- Define the theorem stating the goal of the proof
theorem zero_on_interval {f : ℝ → ℝ} (h : ContinuousFunction f) :
  ∀ x ∈ set.Icc (-1:ℝ) 1, f x = 0 :=
  by sorry

end zero_on_interval_l508_508226


namespace multiples_of_5_with_units_digit_0_l508_508507

theorem multiples_of_5_with_units_digit_0 (h1 : ∀ n : ℕ, n % 5 = 0 → (n % 10 = 0 ∨ n % 10 = 5))
  (h2 : ∀ m : ℕ, m < 200 → m % 5 = 0) :
  ∃ k : ℕ, k = 19 ∧ (∀ x : ℕ, (x < 200) ∧ (x % 5 = 0) → (x % 10 = 0) → k = (k - 1) + 1) := sorry

end multiples_of_5_with_units_digit_0_l508_508507


namespace arith_seq_a2_a9_l508_508540

variable {a : ℕ → ℤ} (S : ℕ → ℤ)
variable (a₁ d : ℤ)

--arith_seq_sum defines the sum of the first n terms of an arithmetic sequence
def arith_seq_sum (n : ℕ) : ℤ := n * a₁ + n * (n - 1) / 2 * d

-- condition for the specific sum of the first 10 terms
axiom h₁ : arith_seq_sum 10 = 120

-- show that a₂ + a₉ = 24
theorem arith_seq_a2_a9 : a 2 + a 9 = 24 :=
by
  -- Let the arithmetic sequence be defined as: aₙ = a₁ + (n - 1) * d
  have h₂ : ∀ n, a n = a₁ + (n - 1) * d := sorry
  -- We are given that arith_seq_sum 10 = 120
  -- arith_seq_sum 10 = 10 * a₁ + 45 * d = 120
  have h₃ : 2 * a₁ + 9 * d = 24 :=
  begin
    -- proof from h₁
    sorry
  end
  -- Substitute values to show a 2 + a 9 = 24
  calc
    a 2 + a 9 = (a₁ + d) + (a₁ + 8 * d) : by rw [h₂ 2, h₂ 9]
          ... = 2 * a₁ + 9 * d : by ring
          ... = 24 : by rw [h₃]

end arith_seq_a2_a9_l508_508540


namespace intersection_one_point_l508_508346

def quadratic_function (x : ℝ) : ℝ := -x^2 + 5 * x
def linear_function (x : ℝ) (t : ℝ) : ℝ := -3 * x + t
def quadratic_combined_function (x : ℝ) (t : ℝ) : ℝ := x^2 - 8 * x + t

theorem intersection_one_point (t : ℝ) : 
  (64 - 4 * t = 0) → t = 16 :=
by
  intro h
  sorry

end intersection_one_point_l508_508346


namespace sum_of_repeating_decimals_l508_508046

theorem sum_of_repeating_decimals :
  let x := 0.4444... in
  let y := 0.3535... in
  (x = 4 / 9) ∧ (y = 35 / 99) →
  x + y = 79 / 99 :=
by
  intros x y hx hy,
  sorry

end sum_of_repeating_decimals_l508_508046


namespace evaluated_sum_approx_2024_l508_508060

def numerator := (∑ i in Finset.range (2023 + 1), (2024 - i) / i)
def denominator := (∑ i in Finset.range (2024 - 1), 1 / (i + 2))

theorem evaluated_sum_approx_2024 :
  (numerator / denominator) = 2024 - (1 / denominator) :=
by { sorry }

end evaluated_sum_approx_2024_l508_508060


namespace stock_yield_percentage_l508_508684

noncomputable def FaceValue : ℝ := 100
noncomputable def AnnualYield : ℝ := 0.20 * FaceValue
noncomputable def MarketPrice : ℝ := 166.66666666666669
noncomputable def ExpectedYieldPercentage : ℝ := 12

theorem stock_yield_percentage :
  (AnnualYield / MarketPrice) * 100 = ExpectedYieldPercentage :=
by
  -- given conditions directly from the problem
  have h1 : FaceValue = 100 := rfl
  have h2 : AnnualYield = 0.20 * FaceValue := rfl
  have h3 : MarketPrice = 166.66666666666669 := rfl
  
  -- we are proving that the yield percentage is 12%
  sorry

end stock_yield_percentage_l508_508684


namespace trapezoid_area_l508_508715

theorem trapezoid_area (h : ℝ) : let b1 := 4 * h; let b2 := 5 * h in (1 / 2) * (b1 + b2) * h = (9 * h^2) / 2 :=
by
  let b1 := 4 * h
  let b2 := 5 * h
  have : (1 / 2) * (b1 + b2) * h = (1 / 2) * (4 * h + 5 * h) * h, by sorry
  have : (1 / 2) * (4 * h + 5 * h) * h = (1 / 2) * 9 * h * h, by sorry
  have : (1 / 2) * 9 * h * h = (9 * h^2) / 2, by sorry
  exact this

end trapezoid_area_l508_508715


namespace part1_tangent_line_at_1_part2_monotonic_intervals_part3_range_of_a_l508_508977

noncomputable def f (x : ℝ) : ℝ := Real.log x
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := a * x + 1
noncomputable def F (x : ℝ) (a : ℝ) : ℝ := f x - g x a

-- (Ⅰ) Equation of the tangent line to y = f(x) at x = 1
theorem part1_tangent_line_at_1 : ∀ x, (f 1 + (1 / 1) * (x - 1)) = x - 1 := sorry

-- (Ⅱ) Intervals where F(x) is monotonic
theorem part2_monotonic_intervals (a : ℝ) : 
  (a ≤ 0 → ∀ x > 0, F x a > 0) ∧ 
  (a > 0 → (∀ x > 0, x < (1 / a) → F x a > 0) ∧ (∀ x > 1 / a, F x a < 0)) := sorry

-- (Ⅲ) Range of a for which f(x) is below g(x) for all x > 0
theorem part3_range_of_a (a : ℝ) : (∀ x > 0, f x < g x a) ↔ a ∈ Set.Ioi (Real.exp (-2)) := sorry

end part1_tangent_line_at_1_part2_monotonic_intervals_part3_range_of_a_l508_508977


namespace exists_k_for_all_n_l508_508256

theorem exists_k_for_all_n (n : ℕ) (hn : 0 < n) :
  ∃ k : ℕ, (0 < k) ∧ (∀ m : ℕ, (1 ≤ m ∧ m ≤ n) → (∀ d : ℕ, (d ≤ 9) → d ∈ digits 10 (m * k))) :=
sorry

end exists_k_for_all_n_l508_508256


namespace smallest_prime_more_than_perfect_square_l508_508319

theorem smallest_prime_more_than_perfect_square : ∃ n : ℕ, nat.prime (n^2 + 20) ∧ (
  ∀ m : ℕ, nat.prime (m^2 + 20) → n^2 + 20 ≤ m^2 + 20) :=
sorry

end smallest_prime_more_than_perfect_square_l508_508319


namespace chord_line_equation_l508_508822

variable (x y : ℝ)
def ellipse : Prop := (x^2/6) + (y^2/5) = 1

def point_P : Prop := P = (2, -1)
def midpoint_condition (x1 y1 x2 y2: ℝ) : Prop := 
  x1 + x2 = 4 ∧ y1 + y2 = -2

theorem chord_line_equation 
  (P : ℝ × ℝ) 
  (A1 A2 : ℝ × ℝ) 
  (hP : point_P P)
  (hA1 : ellipse A1.1 A1.2)
  (hA2 : ellipse A2.1 A2.2)
  (hmid : midpoint_condition A1.1 A1.2 A2.1 A2.2) :
  5 * P.1 - 3 * P.2 - 13 = 0 := 
sorry

end chord_line_equation_l508_508822


namespace cube_surface_area_l508_508330

theorem cube_surface_area (edge : ℝ) (h : edge = 11) : 
  6 * (edge * edge) = 726 := 
by 
  rw h 
  sorry

end cube_surface_area_l508_508330


namespace blake_change_l508_508747

def cost_oranges : ℕ := 40
def cost_apples : ℕ := 50
def cost_mangoes : ℕ := 60
def initial_money : ℕ := 300

def total_cost : ℕ := cost_oranges + cost_apples + cost_mangoes
def change : ℕ := initial_money - total_cost

theorem blake_change : change = 150 := by
  sorry

end blake_change_l508_508747


namespace boatsRUs_total_kayaks_l508_508036

noncomputable def geometric_sum (a r n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

theorem boatsRUs_total_kayaks : 
  let a := 5 in
  let r := 3 in
  let n := 6 in
  geometric_sum a r n = 1820 :=
by
  have a := 5
  have r := 3
  have n := 6
  have S := geometric_sum a r n
  sorry

end boatsRUs_total_kayaks_l508_508036


namespace cos_36_eq_l508_508446

-- Given conditions
def sin_108_eq : Real :=
  sin (108 * (π / 180)) = 3 * sin (36 * (π / 180)) - 4 * (sin (36 * (π / 180)))^3

-- Goal
theorem cos_36_eq : sin_108_eq → cos (36 * (π / 180)) = (1 + Real.sqrt 5) / 4 :=
by
    intro h
    sorry

end cos_36_eq_l508_508446


namespace simplify_exponents_product_l508_508324

theorem simplify_exponents_product :
  (10^0.5) * (10^0.25) * (10^0.15) * (10^0.05) * (10^1.05) = 100 := by
sorry

end simplify_exponents_product_l508_508324


namespace total_cases_sold_is_correct_l508_508303

-- Define the customer groups and their respective number of cases bought
def n1 : ℕ := 8
def k1 : ℕ := 3
def n2 : ℕ := 4
def k2 : ℕ := 2
def n3 : ℕ := 8
def k3 : ℕ := 1

-- Define the total number of cases sold
def total_cases_sold : ℕ := n1 * k1 + n2 * k2 + n3 * k3

-- The proof statement that the total cases sold is 40
theorem total_cases_sold_is_correct : total_cases_sold = 40 := by
  -- Proof content will be provided here.
  sorry

end total_cases_sold_is_correct_l508_508303


namespace initial_rows_l508_508696

theorem initial_rows (r T : ℕ) (h1 : T = 42 * r) (h2 : T = 28 * (r + 12)) : r = 24 :=
by
  sorry

end initial_rows_l508_508696


namespace plane_arrival_time_l508_508352

-- Define the conditions
def departure_time := 11 -- common departure time in hours (11:00)
def bus_speed := 100 -- bus speed in km/h
def train_speed := 300 -- train speed in km/h
def plane_speed := 900 -- plane speed in km/h
def bus_arrival := 20 -- bus arrival time in hours (20:00)
def train_arrival := 14 -- train arrival time in hours (14:00)

-- Given these conditions, we need to prove the plane arrival time
theorem plane_arrival_time : (departure_time + (900 / plane_speed)) = 12 := by
  sorry

end plane_arrival_time_l508_508352


namespace coefficient_z3_l508_508417

noncomputable def P (z : ℤ) : ℤ := 3*z^3 + 2*z^2 - 4*z - 1
noncomputable def Q (z : ℤ) : ℤ := 4*z^4 + z^3 - 2*z^2 + 3

theorem coefficient_z3 :
  (P * Q).coeff 3 = 17 := by
  sorry

end coefficient_z3_l508_508417


namespace arithmetic_sequence_sum_of_b_l508_508458

noncomputable def a : ℕ → ℕ
| 1 => 3
| 2 => 5
| (n+1) => a n + 2

def S : ℕ → ℕ
| 0 => 0
| 1 => a 1
| (n+1) => S n + a (n+1)

def b (n : ℕ) : ℚ := (a n : ℚ) / (3 ^ n)

def T : ℕ → ℚ
| 0 => 0
| (n+1) => T n + b (n+1)

theorem arithmetic_sequence : ∀ n ≥ 2, a n - a (n-1) = 2 :=
by
  intro n h
  induction n with
  | zero => sorry
  | succ k ih =>
      cases k with
      | zero => sorry
      | succ l =>
          have h₁ : a (l+1) - a l = 2 by sorry
          have h₂ : a (l+2) - a (l+1) = a (l+2) - (a l + 2) by sorry
          ...

theorem sum_of_b (n : ℕ) : T n = 2 - (n+2) / (3 ^ n) :=
by
  induction n with
  | zero => sorry
  | succ k ih =>
      have h₁ : T (k+1) = T k + b (k+1) by sorry
      have h₂ : T k = 2 - (k+2) / (3 ^ k) by sorry
      ...

end arithmetic_sequence_sum_of_b_l508_508458


namespace red_black_probability_l508_508702

-- Define the number of cards and ranks
def num_cards : ℕ := 64
def num_ranks : ℕ := 16

-- Define the suits and their properties
def suits := 6
def red_suits := 3
def black_suits := 3
def cards_per_suit := num_ranks

-- Define the number of red and black cards
def red_cards := red_suits * cards_per_suit
def black_cards := black_suits * cards_per_suit

-- Prove the probability that the top card is red and the second card is black
theorem red_black_probability : 
  (red_cards * black_cards) / (num_cards * (num_cards - 1)) = 3 / 4 := by 
  sorry

end red_black_probability_l508_508702


namespace integer_count_between_sqrt8_and_sqrt78_l508_508150

theorem integer_count_between_sqrt8_and_sqrt78 :
  ∃ (n : ℕ), n = 6 ∧ ∀ (x : ℤ), (⌈Real.sqrt 8⌉ ≤ x ∧ x ≤ ⌊Real.sqrt 78⌋) ↔ (3 ≤ x ∧ x ≤ 8) := by
  sorry

end integer_count_between_sqrt8_and_sqrt78_l508_508150


namespace rounded_result_l508_508993

noncomputable def sum : ℝ := 68.257 + 14.0039
noncomputable def product : ℝ := sum * 3

def round_to_hundredth (x : ℝ) : ℝ :=
  let shifted := x * 100
  if (shifted - shifted.floor) < 0.5 then
    shifted.floor / 100
  else
    (shifted.floor + 1) / 100

theorem rounded_result :
  (round_to_hundredth product) = 246.79 :=
by
  sorry

end rounded_result_l508_508993


namespace extreme_value_of_f_range_of_a_l508_508121

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (Real.log x + a) / x
def g (x : ℝ) : ℝ := 1

-- Proof problem (1): Finding the maximum value of f(x)
theorem extreme_value_of_f (a : ℝ) : ∃ (x_max : ℝ), (x_max = Real.exp (1 - a)) ∧ (f x_max a = Real.exp (a - 1)) :=
sorry

-- Proof problem (2): Finding the range of values for a such that the graph of f intersects g = 1 in (0, e^2]
theorem range_of_a (a : ℝ) : (∃ x in Set.Ioc 0 (Real.exp 2), f x a = 1) ↔ (a ≥ 1) :=
sorry

end extreme_value_of_f_range_of_a_l508_508121


namespace polar_to_rectangular_l508_508011

theorem polar_to_rectangular (x y : ℝ)
  (h1 : x = 8)
  (h2 : y = 6)
  (r := Real.sqrt (x^2 + y^2))
  (θ := Real.arctan (y / x))
  (r3 := r^3)
  (θ3 := 3 * θ)
  (cos3θ := 4 * (Real.cos θ)^3 - 3 * Real.cos θ)
  (sin3θ := 3 * Real.sin θ - 4 * (Real.sin θ)^3) :
  (r3 * cos3θ = -352) ∧ (r3 * sin3θ = 936) :=
by 
  rw [h1, h2]
  have hr : r = 10 := by 
    simp [Real.sqrt_eq_rpow]
    norm_num
  rw ←hr at r3
  norm_num at r3
  have hθ : θ = Real.arctan 0.75 := by 
    rw [div_eq_mul_inv, ←real.arctan_div, mul_inv, mul_comm]
    norm_num
  rw ←hθ at θ3
  have cosθ : Real.cos θ = 0.8 := by
    rw hθ
    simp [Real.cos_arctan_of_nonneg, div_eq_one_div]
    norm_num
  have sinθ : Real.sin θ = 0.6 := by
    rw hθ
    simp [Real.sin_arctan_of_nonneg, div_eq_one_div]
    norm_num
  rw [cosθ, sinθ] at cos3θ sin3θ
  simp [cos3θ, sin3θ, r3]
  exact ⟨by norm_num, by norm_num⟩

end polar_to_rectangular_l508_508011


namespace part1_part2_l508_508491

def y (x : ℝ) : ℝ := -x^2 + 8*x - 7

-- Part (1) Lean statement
theorem part1 : ∀ x : ℝ, x < 4 → y x < y (x + 1) := sorry

-- Part (2) Lean statement
theorem part2 : ∀ x : ℝ, (x < 1 ∨ x > 7) → y x < 0 := sorry

end part1_part2_l508_508491


namespace ratio_Sandy_to_Molly_l508_508254

-- Definitions from the given conditions
def Sandy_age_in_6_years : ℕ := 66
def years_into_future : ℕ := 6
def Molly_current_age : ℕ := 45

-- The resulting proof statement, ensuring the problem statement matches:
theorem ratio_Sandy_to_Molly 
  (h1: Sandy_age_in_6_years = 66)
  (h2: years_into_future = 6)
  (h3: Molly_current_age = 45) :
  let Sandy_current_age := Sandy_age_in_6_years - years_into_future in
  (Sandy_current_age : ℚ) / (Molly_current_age : ℚ) = 4 / 3 :=
by 
  sorry

end ratio_Sandy_to_Molly_l508_508254


namespace profit_percent_l508_508660

theorem profit_percent
  (P C : ℝ)                   -- Define P as selling price and C as cost price
  (h : (2/3) * P = 0.88 * C)  -- Given condition
  : ((P - C) / C) * 100 = 32 := 
by 
  have p_eq : P = 1.32 * C := by 
    calc 
      P = (2/3) * P / (2/3) : by sorry
      ... = 0.88 * C / (2/3) : by rw [h]
      ... = 1.32 * C : by sorry
  calc 
    ((P - C) / C) * 100 = ((1.32 * C - C) / C) * 100 : by rw [p_eq]
    ... = ((0.32) * C / C) * 100 : by sorry
    ... = (0.32) * 100 : by sorry
    ... = 32 : by norm_num

end profit_percent_l508_508660


namespace points_lie_on_line_l508_508806

theorem points_lie_on_line (t : ℝ) (ht : t ≠ 0) :
  (let x := (2 * t + 3) / t in
   let y := (2 * t - 3) / t in
   x + y = 4) :=
by
  sorry

end points_lie_on_line_l508_508806


namespace white_balls_count_l508_508185

theorem white_balls_count (T : ℕ) (h1 : 3 / T = 0.10) : T - 3 = 27 :=
by 
  -- proof steps go here, but we're using sorry to skip the proof.
  sorry

end white_balls_count_l508_508185


namespace find_income_of_deceased_l508_508667
noncomputable def income_of_deceased_member 
  (members_before : ℕ) (avg_income_before : ℕ) 
  (members_after : ℕ) (avg_income_after : ℕ) : ℕ :=
  (members_before * avg_income_before) - (members_after * avg_income_after)

theorem find_income_of_deceased 
  (members_before avg_income_before members_after avg_income_after : ℕ) :
  income_of_deceased_member 4 840 3 650 = 1410 :=
by
  -- Problem claims income_of_deceased_member = Income before - Income after
  sorry

end find_income_of_deceased_l508_508667


namespace smallest_b_exists_l508_508797

theorem smallest_b_exists :
  ∃ b : ℕ, (∀ r s : ℤ, r * s = 4032 ∧ r + s = b) ∧
    (∀ b' : ℕ, (∀ r' s' : ℤ, r' * s' = 4032 ∧ r' + s' = b') → b ≤ b') :=
sorry

end smallest_b_exists_l508_508797


namespace base_10_uniqueness_l508_508542

theorem base_10_uniqueness : 
  (∀ a : ℕ, 12 = 3 * 4 ∧ 56 = 7 * 8 ↔ (a * b + (a + 1) = (a + 2) * (a + 3))) → b = 10 :=
sorry

end base_10_uniqueness_l508_508542


namespace b_n_general_formula_T_n_sum_formula_l508_508101

-- Definitions provided by the problem conditions
def a₁ : ℕ := 1
def d : ℕ := 1

def a_n (n : ℕ) : ℕ := a₁ + (n - 1) * d
def S_n (n : ℕ) : ℕ := n * (a₁ + (n-1) * d) / 2

def b_n (n : ℕ) : ℚ := 1 / S_n n
def T_n (n : ℕ) : ℚ := ∑ i in Finset.range n, b_n (i + 1)

-- Proving the questions
theorem b_n_general_formula (n : ℕ) (h : n > 0) : b_n n = 2 / (n * (n + 1)) :=
sorry
  
theorem T_n_sum_formula (n : ℕ) (h : n > 0) : T_n n = 2 * n / (n + 1) :=
sorry

end b_n_general_formula_T_n_sum_formula_l508_508101


namespace sales_second_month_l508_508000

theorem sales_second_month (sale1 sale3 sale4 sale5 sale6 average_sale : ℕ) 
  (h_sale1 : sale1 = 6435) 
  (h_sale3 : sale3 = 6855) 
  (h_sale4 : sale4 = 7230) 
  (h_sale5 : sale5 = 6562) 
  (h_sale6 : sale6 = 7991) 
  (h_avg : average_sale = 7000) : 
  (sale1 + sale3 + sale4 + sale5 + sale6) + x = 6 * average_sale :=
begin
  sorry
end

end sales_second_month_l508_508000


namespace probability_X_eq_Y_l508_508721

theorem probability_X_eq_Y :
  let pairs := { (x, y) : ℝ × ℝ | cos cos x = cos cos y ∧ -5 * π ≤ x ∧ x ≤ 5 * π ∧ -5 * π ≤ y ∧ y ≤ 5 * π } in
  let total_points := 11 * 11 in  -- There are 11 integer multiples of π from −5π to 5π in both x and y.
  let event_points := 11 in       -- Points where x = y happen 11 times in this range.
  (event_points : ℝ) / total_points = 11 / 100 :=
sorry

end probability_X_eq_Y_l508_508721


namespace convex_octagon_quadrilaterals_l508_508048

/-- Given a convex octagon, eight quadrilaterals can be formed by connecting its diagonals. 
Prove that: 
(a) It is possible to have exactly four quadrilaterals that can be inscribed in a circle.
(b) It is impossible to have exactly five quadrilaterals that can be inscribed in a circle. -/
theorem convex_octagon_quadrilaterals (Oct : Type) [ConvexOctagon Oct] :
  (∃ quads : List (Quadrilateral Oct), 
    (quads.length = 8) ∧ 
    (count_quadrilaterals_with_circumcircle quads = 4)) ∧ 
  ¬(∃ quads : List (Quadrilateral Oct), 
    (quads.length = 8) ∧ 
    (count_quadrilaterals_with_circumcircle quads = 5)) :=
sorry

end convex_octagon_quadrilaterals_l508_508048


namespace maximum_profit_and_price_range_l508_508374

-- Definitions
def cost_per_item : ℝ := 60
def max_profit_percentage : ℝ := 0.45
def sales_volume (x : ℝ) : ℝ := -x + 120
def profit (x : ℝ) : ℝ := sales_volume x * (x - cost_per_item)

-- The main theorem
theorem maximum_profit_and_price_range :
  (∃ x : ℝ, x = 87 ∧ profit x = 891) ∧
  (∀ x : ℝ, profit x ≥ 500 ↔ (70 ≤ x ∧ x ≤ 110)) :=
by
  sorry

end maximum_profit_and_price_range_l508_508374


namespace max_FM_l508_508899

noncomputable def maxF (M : ℕ) : ℕ :=
  let a := M / 1000
  let b := (M % 1000) / 100
  let c := (M % 100) / 10
  let d := M % 10
  if b + c + d = 12 ∧ a = b - d ∧ (F(M) / 9).isInt then
    9
  else
    sorry -- other cases

-- Definitions of M, N, F
def F (M : ℕ) : ℤ :=
  let a := M / 1000
  let b := (M % 1000) / 100
  let c := (M % 100) / 10
  let d := M % 10
  let N := 1000 * b + 100 * a + 10 * d + c
  (M - N : ℤ) / 9

-- The problem statement in Lean 4
theorem max_FM (M : ℕ) (h1 : ∃ a b c d : ℕ, M = 1000 * a + 100 * b + 10 * c + d ∧ b + c + d = 12 ∧ a = b - d)
  (h2 : (F M / 9).isInt) : maxF M = 9 := by
  sorry

end max_FM_l508_508899


namespace correct_subtraction_result_l508_508024

theorem correct_subtraction_result (abcde edcba : ℕ) (h : edcba = nat.rev_nat 5 abcde) :
  (edcba - abcde = 34056) :=
sorry

end correct_subtraction_result_l508_508024


namespace roof_shingle_length_l508_508372

theorem roof_shingle_length (width area : ℕ) (h_width : width = 7) (h_area : area = 70) :
  (area / width = 10) :=
by
  rw [h_width, h_area]
  norm_num

end roof_shingle_length_l508_508372


namespace no_solutions_cryptarithm_l508_508673

theorem no_solutions_cryptarithm : 
  ∀ (K O P H A B U y C : ℕ), 
  K ≠ O ∧ K ≠ P ∧ K ≠ H ∧ K ≠ A ∧ K ≠ B ∧ K ≠ U ∧ K ≠ y ∧ K ≠ C ∧ 
  O ≠ P ∧ O ≠ H ∧ O ≠ A ∧ O ≠ B ∧ O ≠ U ∧ O ≠ y ∧ O ≠ C ∧ 
  P ≠ H ∧ P ≠ A ∧ P ≠ B ∧ P ≠ U ∧ P ≠ y ∧ P ≠ C ∧ 
  H ≠ A ∧ H ≠ B ∧ H ≠ U ∧ H ≠ y ∧ H ≠ C ∧ 
  A ≠ B ∧ A ≠ U ∧ A ≠ y ∧ A ≠ C ∧ 
  B ≠ U ∧ B ≠ y ∧ B ≠ C ∧ 
  U ≠ y ∧ U ≠ C ∧ 
  y ≠ C ∧
  K < O ∧ O < P ∧ P > O ∧ O > H ∧ H > A ∧ A > B ∧ B > U ∧ U > P ∧ P > y ∧ y > C → 
  false :=
sorry

end no_solutions_cryptarithm_l508_508673


namespace Jill_water_volume_l508_508956

theorem Jill_water_volume 
  (n : ℕ) (h₀ : 3 * n = 48) :
  n * (1 / 4) + n * (1 / 2) + n * 1 = 28 := 
by 
  sorry

end Jill_water_volume_l508_508956


namespace isosceles_if_any_two_ratios_integers_l508_508201

variable {A B C D E F : Type*}
variable [Triangle A B C] [Points D E F between (B, C) (C, A) (A, B)]
variable {BD DC CE EA AF FB BF FA AE EC CD DB : ℝ}
variable [Circumcenter AD] [Circumcenter BE] [Circumcenter CF]

theorem isosceles_if_any_two_ratios_integers 
  (hacute : acute_triangle A B C)
  (h1 : BD / DC ∈ ℤ ∨ DC / BD ∈ ℤ)
  (h2 : CE / EA ∈ ℤ ∨ EA / CE ∈ ℤ)
  (h3 : AF / FB ∈ ℤ ∨ FB / AF ∈ ℤ)
  (h4 : BF / FA ∈ ℤ ∨ FA / BF ∈ ℤ)
  (h5 : AE / EC ∈ ℤ ∨ EC / AE ∈ ℤ)
  (h6 : CD / DB ∈ ℤ ∨ DB / CD ∈ ℤ)
  : isosceles_triangle A B C :=
sorry

end isosceles_if_any_two_ratios_integers_l508_508201


namespace radius_of_semi_circle_on_EF_l508_508653

-- Definitions based on the given conditions
def is_right_triangle (D E F : ℝ) := D^2 + E^2 = F^2
def semi_circle_area (r : ℝ) := (1 / 2) * π * r^2
def semi_circle_arc_length (r : ℝ) := π * r

-- The main proof problem
theorem radius_of_semi_circle_on_EF (DE DF : ℝ)
  (h1 : semi_circle_area (DE / 2) = 12.5 * π)
  (h2 : semi_circle_arc_length (DF / 2) = 7 * π)
  (h3 : is_right_triangle DE DF (DE^2 + DF^2))
  : (√((DE^2 + DF^2)) / 2) = √74 :=
by
  sorry

end radius_of_semi_circle_on_EF_l508_508653


namespace parallel_lines_m_l508_508987

theorem parallel_lines_m (m : ℝ) :
  (∀ (x y : ℝ), 2 * x + (m + 1) * y + 4 = 0) ∧ (∀ (x y : ℝ), m * x + 3 * y - 2 = 0) →
  (m = -3 ∨ m = 2) :=
by
  sorry

end parallel_lines_m_l508_508987


namespace union_A_B_complement_U_A_intersection_B_range_of_a_l508_508833

-- Define the sets A, B, C, and U
def setA (x : ℝ) : Prop := 2 ≤ x ∧ x ≤ 8
def setB (x : ℝ) : Prop := 1 < x ∧ x < 6
def setC (a : ℝ) (x : ℝ) : Prop := x > a
def U (x : ℝ) : Prop := True  -- U being the universal set of all real numbers

-- Define complements and intersections
def complement (A : ℝ → Prop) (x : ℝ) : Prop := ¬ A x
def intersection (A B : ℝ → Prop) (x : ℝ) : Prop := A x ∧ B x
def union (A B : ℝ → Prop) (x : ℝ) : Prop := A x ∨ B x

-- Proof problems
theorem union_A_B : ∀ x, union setA setB x ↔ (1 < x ∧ x ≤ 8) :=
by 
  intros x
  sorry

theorem complement_U_A_intersection_B : ∀ x, intersection (complement setA) setB x ↔ (1 < x ∧ x < 2) :=
by 
  intros x
  sorry

theorem range_of_a (a : ℝ) : (∃ x, intersection setA (setC a) x) → a < 8 :=
by
  intros h
  sorry

end union_A_B_complement_U_A_intersection_B_range_of_a_l508_508833


namespace correct_statement_l508_508328

theorem correct_statement :
  (0.1:ℝ)^0.8 < (0.2:ℝ)^0.8 :=
sorry

end correct_statement_l508_508328


namespace sum_of_decimals_l508_508026

-- Defining the specific decimal values as constants
def x : ℝ := 5.47
def y : ℝ := 4.26

-- Noncomputable version for addition to allow Lean to handle real number operations safely
noncomputable def sum : ℝ := x + y

-- Theorem statement asserting the sum of x and y
theorem sum_of_decimals : sum = 9.73 := 
by
  -- This is where the proof would go
  sorry

end sum_of_decimals_l508_508026


namespace polygon_diagonals_with_one_non_connecting_vertex_l508_508012

-- Define the number of sides in the polygon
def num_sides : ℕ := 17

-- Define the formula to calculate the number of diagonals in a polygon
def total_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

-- Define the number of non-connecting vertex to any diagonal
def non_connected_diagonals (n : ℕ) : ℕ :=
  n - 3

-- The theorem to state and prove
theorem polygon_diagonals_with_one_non_connecting_vertex :
  total_diagonals num_sides - non_connected_diagonals num_sides = 105 :=
by
  -- The formal proof would go here
  sorry

end polygon_diagonals_with_one_non_connecting_vertex_l508_508012


namespace cookies_prepared_l508_508389

theorem cookies_prepared (n_people : ℕ) (cookies_per_person : ℕ) (total_cookies : ℕ) 
  (h1 : n_people = 25) (h2 : cookies_per_person = 45) : total_cookies = 1125 :=
by
  sorry

end cookies_prepared_l508_508389


namespace sophomores_in_seminar_l508_508184

theorem sophomores_in_seminar (P Q x y : ℕ)
  (h1 : P + Q = 50)
  (h2 : x = y)
  (h3 : x = (1 / 5 : ℚ) * P)
  (h4 : y = (1 / 4 : ℚ) * Q) :
  P = 22 :=
by
  sorry

end sophomores_in_seminar_l508_508184


namespace distance_intersections_l508_508635

theorem distance_intersections (p q : ℕ) (hpq_coprime : Nat.gcd p q = 1)
  (h_intersections : ∀ y : ℝ, y = 4 → ∃ x : ℝ, 5 * x^2 + 3 * x - 2 = y)
  (h_distance : ∀ (C D : ℝ), C = (-3 + Real.sqrt 129) / 10 → D = (-3 - Real.sqrt 129) / 10 → 
     Real.abs (C - D) = sqrt p / q) :
  p - q = 124 := by
  sorry

end distance_intersections_l508_508635


namespace intersection_of_sets_l508_508137

theorem intersection_of_sets (x : ℝ) :
  (∃ y, y = Real.log(1 - 2 * x) ∧ x ∈ {x | x < 1 / 2}) ∩ (∃ y, y = Real.exp x ∧ x ∈ {x | 0 < Real.exp x}) ↔ x ∈ Ioo 0 (1 / 2) := 
by
  sorry

end intersection_of_sets_l508_508137


namespace eternal_number_max_FM_l508_508902

theorem eternal_number_max_FM
  (a b c d : ℕ)
  (h1 : b + c + d = 12)
  (h2 : a = b - d)
  (h3 : (1000 * a + 100 * b + 10 * c + d) - (1000 * b + 100 * a + 10 * d + c) = 81 * (100 * a - 100 * b + c - d))
  (h4 : ∃ k : ℤ, F(M) = 9 * k) :
  ∃ a b c d : ℕ, 100 * (b - d) - 100 * b + 12 - b - 102 * d = 9 := sorry

end eternal_number_max_FM_l508_508902


namespace cyclist_speed_greater_than_pedestrian_l508_508704

open Real

-- Define the conditions in our problem.
variables (A B : Point)                    -- Positions of cities A and B.
variables (v_pedestrian v_cyclist : ℝ)     -- Speeds of the pedestrian and cyclist.
variables (d_AB : ℝ)                      -- Distance between city A and city B.
variables (t1 : ℝ := 1)                   -- Time when cyclist catches up, 1-hour later.
variables (t_total : ℝ := 4)              -- The complete time duration, 4 hours.

-- Definitions based on the problem conditions.
def pedestrian_distance_traveled : ℝ := v_pedestrian * 4
def cyclist_distance_traveled : ℝ := v_cyclist * (4 - 1) / 3 * ((4-1) + 1)

-- Theorem to prove the speed ratio.
theorem cyclist_speed_greater_than_pedestrian :
  v_cyclist = (5/3) * v_pedestrian :=
by
  have h1 : 4 * v_pedestrian = 4 * v_pedestrian := rfl
  have h2 : (4 - 1) * v_cyclist = (5/4 - 3/4) * v_cyclist := rfl
  have h3 : 5 * v_pedestrian = 5/3 * 3 * v_pedestrian := rfl
  -- Calculation of distance based on time and speed.
  sorry

end cyclist_speed_greater_than_pedestrian_l508_508704


namespace intersection_M_N_l508_508136

noncomputable def M : Set ℝ := {-1, 0, 1}
noncomputable def N : Set ℝ := {x | x * (x - 2) ≤ 0}

theorem intersection_M_N : M ∩ N = {0, 1} :=
by {
  sorry
}

end intersection_M_N_l508_508136


namespace true_proposition_l508_508104

-- Define the condition for p
def p := ∃ φ : ℝ, ∀ x : ℝ, sin (x + φ) = sin (-(x + φ))

-- Define the condition for q
def q := ∀ x : ℝ, cos (2 * x) + 4 * sin x - 3 < 0

-- The theorem to be proved
theorem true_proposition : p ∨ ¬ q :=
by
  -- since p is true and q is false,
  -- this makes p ∨ ¬q true
  sorry

end true_proposition_l508_508104


namespace range_of_function_l508_508087

theorem range_of_function: ∀ x : ℝ, -13 ≤ 5 * real.sin x - 12 * real.cos x ∧ 5 * real.sin x - 12 * real.cos x ≤ 13 := by
  sorry

end range_of_function_l508_508087


namespace polynomial_root_divisibility_l508_508221

theorem polynomial_root_divisibility
  (P : ℚ[X])
  (a_n : ℤ) (a_n_minus_1 : ℤ) (a_1 : ℤ) (a_0 : ℤ)
  (hP : P = a_n • X ^ P.natDegree + a_n_minus_1 • X ^ (P.natDegree - 1) + ... + a_1 • X + a_0)
  (p q : ℤ)
  (coprime_pq : Int.gcd p q = 1)
  (h_root : P.eval (p / q) = 0)
  : q ∣ a_n ∧ p ∣ a_0 :=
by
  sorry

end polynomial_root_divisibility_l508_508221


namespace distinct_values_count_l508_508984

noncomputable def Z_k (k : ℕ) : ℂ := (exp (2 * real.pi * complex.I / 20))^(k-1)

theorem distinct_values_count : 
  finset.card (finset.image (λ k, (Z_k k)^1995) (finset.range 20)) = 4 :=
sorry

end distinct_values_count_l508_508984


namespace curves_intersect_bisector_l508_508141

namespace CurveIntersections

def point (α : Type) := (α × α)
def line (α : Type) := (α → α)

-- Definitions for points M and N
def M : point ℝ := (1, 5/4)
def N : point ℝ := (-4, -5/4)

-- Equation of the perpendicular bisector of MN
def perpendicular_bisector (x : ℝ) : ℝ := -2 * (x + 3/2)

-- Definitions for curves
def curve1 (x y : ℝ) : Prop := 4*x + 2*y - 1 = 0
def curve2 (x y : ℝ) : Prop := x^2 + y^2 = 3
def curve3 (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1
def curve4 (x y : ℝ) : Prop := x^2 / 2 - y^2 = 1

-- The theorem stating the desired result
theorem curves_intersect_bisector :
  ∃ (x y : ℝ), curve2 x y ∧ y = perpendicular_bisector x ∧
  ∃ (x y : ℝ), curve3 x y ∧ y = perpendicular_bisector x ∧
  ∃ (x y : ℝ), curve4 x y ∧ y = perpendicular_bisector x :=
sorry

end CurveIntersections

end curves_intersect_bisector_l508_508141


namespace largest_possible_cos_a_l508_508564

theorem largest_possible_cos_a (a b c : ℝ) (h1 : Real.sin a = Real.cot b) 
  (h2 : Real.sin b = Real.cot c) (h3 : Real.sin c = Real.cot a) : 
  Real.cos a ≤ Real.sqrt ((3 - Real.sqrt 5) / 2) :=
by sorry

end largest_possible_cos_a_l508_508564


namespace range_of_a_l508_508857

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f(x) = x^3 - a*x^2 + 4 ∧ f(x) = 0 → x > 0 → ((∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f(x₁) = 0 ∧ f(x₂) = 0))) → 3 < a :=
begin
  sorry
end

end range_of_a_l508_508857


namespace total_cost_correct_l508_508666

-- Define the costs for each day
def day1_rate : ℝ := 150
def day1_miles_cost : ℝ := 0.50 * 620
def gps_service_cost : ℝ := 10
def day1_total_cost : ℝ := day1_rate + day1_miles_cost + gps_service_cost

def day2_rate : ℝ := 100
def day2_miles_cost : ℝ := 0.40 * 744
def day2_total_cost : ℝ := day2_rate + day2_miles_cost + gps_service_cost

def day3_rate : ℝ := 75
def day3_miles_cost : ℝ := 0.30 * 510
def day3_total_cost : ℝ := day3_rate + day3_miles_cost + gps_service_cost

-- Define the total cost
def total_cost : ℝ := day1_total_cost + day2_total_cost + day3_total_cost

-- Prove that the total cost is equal to the calculated value
theorem total_cost_correct : total_cost = 1115.60 :=
by
  -- This is where the proof would go, but we leave it out for now
  sorry

end total_cost_correct_l508_508666


namespace fault_line_movement_l508_508385

theorem fault_line_movement (total_movement : ℝ) (last_year_movement : ℝ) (current_year_movement : ℝ) 
  (h_total : total_movement = 6.5) (h_last_year : last_year_movement = 5.25) : 
  current_year_movement = 1.25 :=
by
  have h1 : current_year_movement = total_movement - last_year_movement
  { sorry },
  rw [h_total, h_last_year] at h1,
  exact h1

end fault_line_movement_l508_508385


namespace magician_ratio_l508_508008

def magician_problem 
  (total_performances : ℕ)
  (extra_reappearance_ratio : ℤ)
  (total_reapperances : ℤ)
  (extra_performances : ℕ)
  (x : ℤ)
  (expected_reappearances : ℤ) : Prop :=
  (total_performances / extra_reappearance_ratio = extra_performances) →
  (expected_reappearances = total_performances) →
  (total_reapperances = expected_reappearances - x + int.ofNat extra_performances) →
  (x = 10) →
  (x / total_performances = 1 / 10)

theorem magician_ratio
  (total_performances := 100)
  (extra_reappearance_ratio := 5)
  (total_reapperances := 110)
  (extra_performances := 20)
  (x := 10)
  (expected_reappearances := 100) : magician_problem total_performances extra_reappearance_ratio total_reapperances extra_performances x expected_reappearances :=
by
  intro h1 h2 h3 h4
  rw [h4]
  norm_num
  sorry

end magician_ratio_l508_508008


namespace solve_sum_of_B_l508_508771

def B : Set ℕ := { n | ∀ p : ℕ, p.prime → p ∣ n → p = 2 ∨ p = 3 ∨ p = 7 }

noncomputable def sum_reciprocals (s : Set ℕ) : ℚ :=
  ∑' n in s, 1 / n

theorem solve_sum_of_B :
  let S := sum_reciprocals B
  let m := S.num
  let n := S.denom
  ∃ m' n' : ℕ, m' = m ∧ n' = n ∧ Int.gcd m' n' = 1 ∧ m + n = 9 :=
by
  let S := sum_reciprocals B
  let m := S.num
  let n := S.denom
  have h1 : ∀ p : ℕ, p.prime → p = 2 ∨ p = 3 ∨ p = 7
    ∧ Int.gcd m n = 1
    ∧ m + n = 9 := sorry
  exact ⟨m, n, rfl, rfl, h1.2.1, h1.2.2⟩

end solve_sum_of_B_l508_508771


namespace distance_between_parallel_lines_l508_508488

theorem distance_between_parallel_lines :
  ∀ (x y : ℝ), (2 * x + y - 1 = 0) ∧ (2 * x + y + 1 = 0) → 
  let A : ℝ := 2
      B : ℝ := 1
      C₁ : ℝ := -1
      C₂ : ℝ := 1
      d : ℝ := (abs (C₁ - C₂)) / (sqrt (A * A + B * B))
      rational_d := d * (sqrt 5 / sqrt 5) 
  in rational_d = 2 * sqrt 5 / 5 :=
by
  sorry

end distance_between_parallel_lines_l508_508488


namespace find_theta_sum_l508_508774

noncomputable def complex_numbers_sum (n : ℕ) (z : ℂ) (θ : ℕ → ℝ) : ℝ :=
  if 2 * n ∈ Finset.range 24 ∧
     (∀ k, (0 ≤ θ k ∧ θ k < 360)) ∧
     z^36 - z^12 - z^3 - 1 = 0 ∧
     complex.abs z = 1 ∧
     (∀ k, complex.ofReal (cos (θ k) * π / 180) + complex.I * complex.ofReal (sin (θ k) * π / 180) = z)
  then (Finset.range n).sum (λ k, θ (2 * k + 1))
  else 0

theorem find_theta_sum (n : ℕ) (z : ℂ) (θ : ℕ → ℝ) :
  complex_numbers_sum n z θ = 2160 :=
sorry

end find_theta_sum_l508_508774


namespace smoking_related_to_lung_disease_l508_508356

theorem smoking_related_to_lung_disease (n : ℕ) (K_squared : ℝ) (P_critical_95 : ℝ) (P_critical_99 : ℝ) 
  (h1 : n = 11000)
  (h2 : K_squared = 5.231)
  (h3 : P_critical_95 = 3.841)
  (h4 : P_critical_99 = 6.635)
  (null_hypothesis : Prop) :
  P_critical_95 ≤ K_squared ∧ K_squared < P_critical_99 → more_than_95_percent_confidence (¬ null_hypothesis) :=
by
  -- Placeholder for the proof
  sorry

end smoking_related_to_lung_disease_l508_508356


namespace parabola_constant_term_l508_508369

theorem parabola_constant_term (b c : ℝ)
  (h1 : 2 * b + c = 8)
  (h2 : -2 * b + c = -4)
  (h3 : 4 * b + c = 24) :
  c = 2 :=
sorry

end parabola_constant_term_l508_508369


namespace max_x_values_l508_508127

noncomputable def y (x : ℝ) : ℝ := (1/2) * (Real.cos x)^2 + (Real.sqrt 3 / 2) * (Real.sin x) * (Real.cos x) + 1

theorem max_x_values :
  {x : ℝ | ∃ k : ℤ, x = k * Real.pi + Real.pi / 6} = {x : ℝ | y x = y (x)} :=
sorry

end max_x_values_l508_508127


namespace f_monotonically_increasing_f_range_on_interval_l508_508854

noncomputable def f (x : ℝ) : ℝ := 1 - 2 / (2^x + 1)

theorem f_monotonically_increasing : 
  ∀ x1 x2 : ℝ, x1 < x2 → f(x1) < f(x2) :=
begin
  sorry
end

theorem f_range_on_interval : 
  ∀ m : ℝ, 
    (- (3 / 5)) ≤ m ∧ m ≤ (3 / 5) ↔ 
    ∃ x : ℝ, (-2 ≤ x) ∧ (x ≤ 2) ∧ (f(x) = m) :=
begin
  sorry
end

end f_monotonically_increasing_f_range_on_interval_l508_508854


namespace no_right_triangles_with_perimeter_eq_5_times_inradius_l508_508152

theorem no_right_triangles_with_perimeter_eq_5_times_inradius :
  ∀ (a b : ℕ), ∃ (c : ℕ), (a > 0 ∧ b > 0) → c = Int.natAbs (Nat.sqrt (a^2 + b^2)) →
  let perim := a + b + c,
  perim = 5 * (a * b / perim) → false :=
by
  intros a b c hpos hc_peri hr_eq
  sorry

end no_right_triangles_with_perimeter_eq_5_times_inradius_l508_508152


namespace residue_modulo_l508_508972

def S : ℤ := (List.range 2020).map (λ n, if n % 2 = 0 then n + 1 else -(n + 1)).sum

theorem residue_modulo (S : ℤ) : (S % 2020) = 1010 := by
  sorry

end residue_modulo_l508_508972


namespace productivity_increase_correct_l508_508781

def productivity_increase (that: ℝ) :=
  ∃ x : ℝ, (x + 1) * (x + 1) * 2500 = 2809

theorem productivity_increase_correct :
  productivity_increase (0.06) :=
by
  sorry

end productivity_increase_correct_l508_508781


namespace scientific_notation_of_11580000_l508_508926

theorem scientific_notation_of_11580000 :
  11_580_000 = 1.158 * 10^7 :=
sorry

end scientific_notation_of_11580000_l508_508926


namespace spherical_coordinates_l508_508049

theorem spherical_coordinates (x y z : ℝ) (h₁ : x = 1) (h₂ : y = sqrt 3) (h₃ : z = 2) :
  ∃ (ρ θ φ : ℝ), ρ = 3 ∧ θ = π / 3 ∧ φ = Real.arccos (2 / 3) ∧
  (x, y, z) = (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ) :=
by
  use 3, π / 3, Real.arccos (2 / 3)
  split
  rfl
  split
  rfl
  split
  rfl
  sorry

end spherical_coordinates_l508_508049


namespace temperature_on_tuesday_l508_508269

theorem temperature_on_tuesday 
  (M T W Th F Sa : ℝ)
  (h1 : (M + T + W) / 3 = 38)
  (h2 : (T + W + Th) / 3 = 42)
  (h3 : (W + Th + F) / 3 = 44)
  (h4 : (Th + F + Sa) / 3 = 46)
  (hF : F = 43)
  (pattern : M + 2 = Sa ∨ M - 1 = Sa) :
  T = 80 :=
sorry

end temperature_on_tuesday_l508_508269


namespace part_one_part_two_l508_508484

noncomputable def f (x a : ℝ) : ℝ := (x + 1) * Real.log x - a * x + a

theorem part_one (a : ℝ) (h_pos : 0 < a) :
  (∀ x > 0, (Real.log x + 1/x + 1 - a) ≥ 0) ↔ (0 < a ∧ a ≤ 2) :=
sorry

theorem part_two (a : ℝ) (h_pos : 0 < a) :
  (∀ x, (x - 1) * (f x a) ≥ 0) ↔ (0 < a ∧ a ≤ 2) :=
sorry

end part_one_part_two_l508_508484


namespace additional_sugar_needed_l508_508305

theorem additional_sugar_needed : 
  ∀ (required_sugar stored_sugar : ℕ), required_sugar = 450 → stored_sugar = 287 → (required_sugar - stored_sugar) = 163 :=
by 
  intros required_sugar stored_sugar hreq hst
  rw [hreq, hst]
  exact rfl

end additional_sugar_needed_l508_508305


namespace liam_mia_no_savings_l508_508021

/-- A store sells windows at $150 each and offers a discount: two free windows for every nine purchased.
Liam needs ten windows, and Mia needs twelve windows. Prove that they will not save any money if they buy the windows jointly rather than separately. -/
theorem liam_mia_no_savings:
    let price_per_window := 150
    let free_windows_for_nine_purchased := 2
    let liam_windows_needed := 10
    let mia_windows_needed := 12
    let total_windows_needed := liam_windows_needed + mia_windows_needed
    let cost_windows_needed (n : Nat) := (n - n / 11 * free_windows_for_nine_purchased) * price_per_window
in cost_windows_needed total_windows_needed = cost_windows_needed liam_windows_needed + cost_windows_needed mia_windows_needed := 
    sorry

end liam_mia_no_savings_l508_508021


namespace triangle_area_constant_find_circle_equation_l508_508940

theorem triangle_area_constant :
  ∀ (t : ℝ), let x := 2 * t,
                y := 2 * (real.sqrt 3) / t in
  let A := (x, 0),
      B := (0, y) in
  (1/2) * (real.abs x) * (real.abs y) = 2 * (real.sqrt 3) :=
sorry

theorem find_circle_equation :
  let t : ℝ := 1,
      Mx := t,
      My := real.sqrt 3 in
  let d := 2 * (real.sqrt 3 - 1),
      radius := 2 in
  ∀ (l_x l_y : ℝ), l_y = (-real.sqrt 3 / 3) * l_x + 4 →
    ((l_y = My → l_x = Mx) ∧ (2 * radius = d)) →
  (∀ (x y : ℝ), (x - 1) ^ 2 + (y - real.sqrt 3) ^ 2 = 4) :=
sorry

end triangle_area_constant_find_circle_equation_l508_508940


namespace area_EHF_leq_one_fourth_area_ABCD_not_general_statement_for_arbitrary_quadrilateral_l508_508228

open EuclideanGeometry
open Real
open Set

variables {A B C D E F H G : Point}

-- Defining the trapezoid and its properties
def is_trapezoid (A B C D : Point) : Prop :=
  parallel A B C D ∧ ∃ E F H G : Point, 
    (lies_on_segment E A B) ∧ 
    (lies_on_segment F C D) ∧ 
    (intersects_at H C E B F) ∧ 
    (intersects_at G E D A F)

-- The theorem statement for the area of EHF
theorem area_EHF_leq_one_fourth_area_ABCD 
  (A B C D E F H G : Point) :
  is_trapezoid A B C D → 
  (area_triangle E H F) ≤ (1/4 * area_trapezoid A B C D) := 
sorry

-- The theorem statement for an arbitrary convex quadrilateral
theorem not_general_statement_for_arbitrary_quadrilateral 
  (A B C D E F H G : Point) :
  is_convex_quadrilateral A B C D →
  ¬( (area_triangle E H F) ≤ (1/4 * area_quadrilateral A B C D) ) :=
sorry

end area_EHF_leq_one_fourth_area_ABCD_not_general_statement_for_arbitrary_quadrilateral_l508_508228


namespace smallest_m_for_integral_solutions_l508_508659

theorem smallest_m_for_integral_solutions (p q : ℤ) (h : p * q = 42) (h0 : p + q = m / 15) : 
  0 < m ∧ 15 * p * p - m * p + 630 = 0 ∧ 15 * q * q - m * q + 630 = 0 →
  m = 195 :=
by 
  sorry

end smallest_m_for_integral_solutions_l508_508659


namespace total_cases_sold_l508_508298

theorem total_cases_sold : 
  let people := 20 in
  let first_8_cases := 8 * 3 in
  let next_4_cases := 4 * 2 in
  let last_8_cases := 8 * 1 in
  first_8_cases + next_4_cases + last_8_cases = 40 := 
by
  let people := 20
  let first_8_cases := 8 * 3
  let next_4_cases := 4 * 2
  let last_8_cases := 8 * 1
  have h1 : first_8_cases = 24 := by rfl
  have h2 : next_4_cases = 8 := by rfl
  have h3 : last_8_cases = 8 := by rfl
  have h : first_8_cases + next_4_cases + last_8_cases = 24 + 8 + 8 := by rw [h1, h2, h3]
  show 24 + 8 + 8 = 40 from rfl

end total_cases_sold_l508_508298


namespace sum_of_digits_253_l508_508366

-- Define the digits of the number, the representation of the number, and the conditions
variables (a b c : ℕ)

-- Given conditions
def condition1 : Prop := b = a + c
def condition2 : Prop := 100c + 10b + a = 100a + 10b + c + 99
def number253 : Prop := 100a + 10b + c = 253

-- Prove the sum of the digits is 10
theorem sum_of_digits_253 
  (h1 : condition1)  -- b = a + c
  (h2 : condition2)  -- 100c + 10b + a = 100a + 10b + c + 99
  (h3 : number253) : -- 100a + 10b + c = 253
  a + b + c = 10 :=
sorry

end sum_of_digits_253_l508_508366


namespace point_B_coordinates_l508_508826

/-- Given point A at (2, 1) and vector OA rotates clockwise around the origin O by π/2, 
prove that the coordinates of point B are (1, -2). -/
theorem point_B_coordinates :
  let A := (2 : ℤ, 1 : ℤ),
      rotate_clockwise_90 (p : ℤ × ℤ) : ℤ × ℤ := (p.2, -p.1)
  in rotate_clockwise_90 A = (1, -2) :=
by
  sorry

end point_B_coordinates_l508_508826


namespace find_asymptote_slope_l508_508279

-- Definitions
def hyperbola_eq (x y : ℝ) : Prop := (x^2) / 121 - (y^2) / 81 = 1
def asymptotes_eq (n : ℝ) (x y : ℝ) : Prop := y = n * x ∨ y = -n * x

-- Theorem statement
theorem find_asymptote_slope :
  ∀ x y n > 0, hyperbola_eq x y → asymptotes_eq n x y → n = 9 / 11 :=
sorry

end find_asymptote_slope_l508_508279


namespace inequality_proof_l508_508512

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (b + c) / (2 * a) + (c + a) / (2 * b) + (a + b) / (2 * c) ≥ (2 * a) / (b + c) + (2 * b) / (c + a) + (2 * c) / (a + b) :=
by
  sorry

end inequality_proof_l508_508512


namespace jason_borrowed_amount_l508_508193

theorem jason_borrowed_amount : 
  ∀ (babysitting_hours : ℕ) (P : ℕ → ℕ), 
  (∀ n, P n = 
    match n % 8 with
    | 0 => 2
    | 1 => 4
    | 2 => 6
    | 3 => 8
    | 4 => 10
    | 5 => 12
    | 6 => 2
    | _ => 4
    end) 
  → babysitting_hours = 48 
  → ( ∑ i in finset.range babysitting_hours, P i) = 288 :=
by
  intros babysitting_hours P hP hHours,
  sorry

end jason_borrowed_amount_l508_508193


namespace tower_remainder_l508_508690

def T : ℕ := 4374 -- This is the value of T_9 from the solution.

theorem tower_remainder : T % 500 = 374 :=
by {
  -- Since T is defined as a constant 4374, we can directly compute the mod
  have h : T = 4374 := rfl,
  rw h,
  exact Nat.mod_eq_of_lt (Nat.lt_of_succ_lt_succ (Nat.succ_lt_succ (Nat.succ_lt_succ
    (Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.succ_lt_succ 
    (Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.zero_lt_succ 8))))))))))),
}

end tower_remainder_l508_508690


namespace parallel_lines_distance_l508_508133

theorem parallel_lines_distance (c_2 : ℝ) :
  (c_2 = 2 * Real.sqrt 34 - 2 ∨ c_2 = -2 * Real.sqrt 34 - 2) → 
  let y := (x : ℝ) ↦ (5 / 3) * x + c_2 in
  let y2 := (x : ℝ) ↦ (5 / 3) * x - 2 in
  ∃ line : ℝ → ℝ, (line = y ∧ (∀ x, (line x) - y2 x = 6)) :=
by
  sorry

end parallel_lines_distance_l508_508133


namespace percentage_increase_on_friday_l508_508967

theorem percentage_increase_on_friday (avg_books_per_day : ℕ) (friday_books : ℕ) (total_books_per_week : ℕ) (days_open : ℕ)
  (h1 : avg_books_per_day = 40)
  (h2 : total_books_per_week = 216)
  (h3 : days_open = 5)
  (h4 : friday_books > avg_books_per_day) :
  (((friday_books - avg_books_per_day) * 100) / avg_books_per_day) = 40 :=
sorry

end percentage_increase_on_friday_l508_508967


namespace signal_commonality_l508_508929

theorem signal_commonality {n m : ℕ} (h1 : n ≥ 4) (h2 : m ≥ 1)
  (h3 : ∀ {A B C : Fin n}, (A ≠ B ∧ B ≠ C ∧ A ≠ C) → (knowledge A B ∧ knowledge B C ∧ knowledge C A) → signal A B C)
  (h4 : ∀ {A B C D : Fin n}, (A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ A ≠ D) → (signal A B C) → (signal A D ∨ signal B D ∨ signal C D) ≤ 1) :
  ∃ (A B C : Fin n), (knowledge A B ∧ knowledge B C ∧ knowledge C A) ∧
  (∃ (x : ℕ), x ≤ (n + 3 - 18 * m / n) ∧
  (∀ D : Fin n, (knowledge D A ∨ knowledge D B ∨ knowledge D C) → (signal D A ∨ signal D B ∨ signal D C) → x < n)) :=
by sorry

end signal_commonality_l508_508929


namespace magnitude_vector_sum_eq_sqrt13_l508_508872

-- Define the vector a
def a : ℝ × ℝ := (-1, real.sqrt 2)

-- Define the magnitude of vector b
def magnitude_b : ℝ := 2

-- Define the angle between vectors a and b
def theta : ℝ := real.pi / 6  -- 30 degrees in radians

-- The proof statement
theorem magnitude_vector_sum_eq_sqrt13
  (ha : a = (-1, real.sqrt 2))
  (hb : magnitude_b = 2)
  (htheta : theta = real.pi / 6) :
  ∥(⟨-1, real.sqrt 2⟩ : ℝ × ℝ) + (⟨_, _⟩ : ℝ × ℝ)∥ = real.sqrt 13 :=
sorry

end magnitude_vector_sum_eq_sqrt13_l508_508872


namespace problem_1163_prime_and_16424_composite_l508_508875

theorem problem_1163_prime_and_16424_composite :
  let x := 1910 * 10000 + 1112
  let a := 1163
  let b := 16424
  x = a * b →
  Prime a ∧ ¬ Prime b :=
by
  intros h
  sorry

end problem_1163_prime_and_16424_composite_l508_508875


namespace Lloyd_hourly_rate_is_3_5_l508_508236

/-!
Lloyd normally works 7.5 hours per day and earns a certain amount per hour.
For each hour he works in excess of 7.5 hours on a given day, he is paid 1.5 times his regular rate.
If Lloyd works 10.5 hours on a given day, he earns $42 for that day.
-/

variable (Lloyd_hourly_rate : ℝ)  -- regular hourly rate

def Lloyd_daily_earnings (total_hours : ℝ) (regular_hours : ℝ) (hourly_rate : ℝ) : ℝ :=
  let excess_hours := total_hours - regular_hours
  let excess_earnings := excess_hours * (1.5 * hourly_rate)
  let regular_earnings := regular_hours * hourly_rate
  excess_earnings + regular_earnings

-- Given conditions
axiom H1 : 7.5 = 7.5
axiom H2 : ∀ R : ℝ, Lloyd_hourly_rate = R
axiom H3 : ∀ R : ℝ, ∀ excess_hours : ℝ, Lloyd_hourly_rate + excess_hours = 1.5 * R
axiom H4 : Lloyd_daily_earnings 10.5 7.5 Lloyd_hourly_rate = 42

-- Prove Lloyd earns $3.50 per hour.
theorem Lloyd_hourly_rate_is_3_5 : Lloyd_hourly_rate = 3.5 :=
sorry

end Lloyd_hourly_rate_is_3_5_l508_508236


namespace intersect_chord_length_l508_508638

noncomputable def curve_polar := λ (ρ θ : ℝ), ρ = 4 * cos θ

noncomputable def curve_rectangular := λ (x y : ℝ), x^2 + y^2 - 4 * x = 0

noncomputable def line_parametric := λ (x y t : ℝ), (x = 3 + 4 * t ∧ y = 2 + 3 * t)

noncomputable def line_rectangular := λ (x y : ℝ), 3 * x - 4 * y - 1 = 0

theorem intersect_chord_length :
  (∀ ρ θ, curve_polar ρ θ → curve_rectangular ρ (4 * cos θ)) →
  (∀ x y t, line_parametric x y t → line_rectangular x y) →
  (∃ (x y : ℝ), curve_rectangular x y ∧ line_rectangular x y) ∧
  ∃ d : ℝ, d < 2 ∧ 2 * real.sqrt (4 - d^2) = 2 * real.sqrt 3 :=
by
  intro h1 h2
  have : ∀ ρ θ, ρ^2 = 4 * ρ * cos θ, from λ ρ θ h, sorry
  use (2, 0)
  split
  ·apply this
  · use 1
  sorry

end intersect_chord_length_l508_508638


namespace compare_a_b_c_l508_508214

theorem compare_a_b_c :
  let a := (1/2)^(1/3)
  let b := Real.log2 (1/3)
  let c := Real.log2 3
  c > a ∧ a > b :=
by
  sorry

end compare_a_b_c_l508_508214


namespace largest_cos_a_of_angles_l508_508567

noncomputable def solve_for_cos_a (a b c : ℝ) : ℝ :=
  sqrt ((3 - sqrt 5) / 2)

theorem largest_cos_a_of_angles (a b c : ℝ) 
  (h1 : Real.sin a = Real.cot b) 
  (h2 : Real.sin b = Real.cot c)
  (h3 : Real.sin c = Real.cot a) : 
  Real.cos a = solve_for_cos_a a b c := 
  sorry

end largest_cos_a_of_angles_l508_508567


namespace fourth_quadrant_correct_l508_508724

def Point := (ℝ × ℝ)

def is_in_fourth_quadrant (p : Point) : Prop :=
  p.1 > 0 ∧ p.2 < 0

def points : List Point :=
  [(1, 2), (-1, 2), (1, -2), (-1, -2)]

def fourth_quadrant_point (ps : List Point) : Point :=
  ps.filter is_in_fourth_quadrant |> List.head!  -- Unsafe, assuming there's always a valid point

theorem fourth_quadrant_correct {A B C D : Point} (hA : A = (1, 2)) (hB : B = (-1, 2)) (hC : C = (1, -2)) (hD : D = (-1, -2)) :
  (fourth_quadrant_point [A, B, C, D]) = C :=
by {
  rw [fourth_quadrant_point, List.filter, List.head!],
  sorry 
}

end fourth_quadrant_correct_l508_508724


namespace factor_polynomial_roots_l508_508075

noncomputable def quadratic_roots (a b c : ℝ) : set ℝ :=
  { x | a * x^2 + b * x + c = 0 }

theorem factor_polynomial_roots :
  quadratic_roots 10 23 (-7) = { ( -23 + Real.sqrt 809) / 20, ( -23 - Real.sqrt 809) / 20 } :=
by
  sorry

end factor_polynomial_roots_l508_508075


namespace rotated_ln_eq_neg_exp_l508_508630

open Real

theorem rotated_ln_eq_neg_exp (x : ℝ) (h : x > 0) : ∃ y, y = ln x → ∃ x', x' = -x → y = -exp x' := 
sorry

end rotated_ln_eq_neg_exp_l508_508630


namespace total_ants_approx_59_million_l508_508016

def field_width_feet := 250
def field_length_feet := 330
def ants_per_square_inch := 5
def feet_to_inches := 12

def field_width_inches := field_width_feet * feet_to_inches
def field_length_inches := field_length_feet * feet_to_inches
def field_area_square_inches := field_width_inches * field_length_inches
def total_ants := ants_per_square_inch * field_area_square_inches

-- The statement to be proven
theorem total_ants_approx_59_million : total_ants ≈ 59_000_000 := 
by
  -- Provide necessary steps for the theorem proof
  sorry

end total_ants_approx_59_million_l508_508016


namespace total_amount_paid_l508_508552

theorem total_amount_paid (monthly_payment_1 monthly_payment_2 : ℕ) (years_1 years_2 : ℕ)
  (monthly_payment_1_eq : monthly_payment_1 = 300)
  (monthly_payment_2_eq : monthly_payment_2 = 350)
  (years_1_eq : years_1 = 3)
  (years_2_eq : years_2 = 2) :
  let annual_payment_1 := monthly_payment_1 * 12
  let annual_payment_2 := monthly_payment_2 * 12
  let total_1 := annual_payment_1 * years_1
  let total_2 := annual_payment_2 * years_2
  total_1 + total_2 = 19200 :=
by
  sorry

end total_amount_paid_l508_508552


namespace total_cases_of_cat_food_sold_l508_508295

theorem total_cases_of_cat_food_sold :
  (let first_eight := 8 * 3 in
   let next_four := 4 * 2 in
   let last_eight := 8 * 1 in
   first_eight + next_four + last_eight = 40) :=
by
  -- Given conditions:
  -- first_8_customers: 8 customers bought 3 cases each
  -- second_4_customers: 4 customers bought 2 cases each
  -- last_8_customers: 8 customers bought 1 case each
  let first_eight := 8 * 3
  let next_four := 4 * 2
  let last_eight := 8 * 1
  -- Sum of all cases
  show first_eight + next_four + last_eight = 40
  sorry

end total_cases_of_cat_food_sold_l508_508295


namespace abc_zero_l508_508969

theorem abc_zero {a b c : ℝ} 
(h1 : (a + b) * (b + c) * (c + a) = a * b * c)
(h2 : (a^3 + b^3) * (b^3 + c^3) * (c^3 + a^3) = a^3 * b^3 * c^3) : 
a * b * c = 0 := 
by sorry

end abc_zero_l508_508969


namespace find_angle_A_find_side_c_l508_508761

theorem find_angle_A 
  (a b c A B C : ℝ)
  (h1 : b = a * cos C + 1 / 2 * c ∨ sin^2 A - (sin B - sin C)^2 = 2 * sin B * sin (C + π / 3) - sqrt 3 * sin B * cos C)
  (h2 : A + B + C = π) :
  A = π / 3 :=
sorry

theorem find_side_c 
  (b c : ℝ) 
  (h1 : b = 2)
  (h2 : b * cos C + c * cos B = sqrt 7)
  (A B C : ℝ)
  (h3 : A + B + C = π)
  (h4 : A = π / 3) :
  c = 3 :=
sorry

end find_angle_A_find_side_c_l508_508761


namespace real_solutions_to_equation_l508_508079

/-- Prove that the solutions to the equation x - 3 * {x} - {3 * {x}} = 0 are exactly the set 
    {0, 6 / 5, 7 / 5, 13 / 5, 14 / 5}, given that {a} represents the fractional part of a. -/
theorem real_solutions_to_equation :
  {x : ℝ | x - 3 * fract x - fract (3 * fract x) = 0} = {0, 6/5, 7/5, 13/5, 14/5} :=
by
  sorry

end real_solutions_to_equation_l508_508079


namespace line_equation_l508_508624

theorem line_equation (x y : ℝ) (h : ∀ x : ℝ, (x - 2) * 1 = y) : x - y - 2 = 0 :=
sorry

end line_equation_l508_508624


namespace range_of_ab_l508_508450

def circle_eq (x y : ℝ) := x^2 + y^2 + 2 * x - 4 * y + 1 = 0
def line_eq (a b x y : ℝ) := 2 * a * x - b * y + 2 = 0

theorem range_of_ab (a b : ℝ) :
  (∃ x y : ℝ, circle_eq x y ∧ line_eq a b x y) ∧ (∃ x y : ℝ, x = -1 ∧ y = 2) →
  ab <= 1/4 := 
by
  sorry

end range_of_ab_l508_508450


namespace vector_dot_product_l508_508143

open Real

variables (a b : ℝ × ℝ)

def condition1 : Prop := (a.1 + b.1 = 1 ∧ a.2 + b.2 = -3)
def condition2 : Prop := (a.1 - b.1 = 3 ∧ a.2 - b.2 = 7)
def dot_product : ℝ := a.1 * b.1 + a.2 * b.2

theorem vector_dot_product :
  condition1 a b ∧ condition2 a b → dot_product a b = -12 := by
  sorry

end vector_dot_product_l508_508143


namespace joyce_previous_property_size_l508_508964

theorem joyce_previous_property_size
  (new_property_is_10_times_prev : ∀ (prev : ℝ), ℝ → ℝ := λ prev new, new = 10 * prev)
  (pond_size : ℝ := 1)
  (suitable_land : ℝ := 19)
  (total_new_property : ℝ := suitable_land + pond_size) :
  ∃ prev : ℝ, total_new_property = 10 * prev ∧ prev = 2 :=
by
  sorry

end joyce_previous_property_size_l508_508964


namespace smallest_sum_l508_508583

theorem smallest_sum : 
  ∀ (x : Fin 100 → ℕ), 
  (∀ i, x i > 1) →
  let table := λ i j : Fin 100, real.log (x i : ℝ) / real.log (x j : ℝ) - 2 in
  ∑ i in Finset.range 100, ∑ j in Finset.range 100, table i j = -10000 :=
begin
  intros x hx,
  let table := λ i j : Fin 100, real.log (x i : ℝ) / real.log (x j : ℝ) - 2,
  sorry
end

end smallest_sum_l508_508583


namespace jenna_average_speed_l508_508194

theorem jenna_average_speed (total_distance : ℕ) (total_time : ℕ) 
(first_segment_speed : ℕ) (second_segment_speed : ℕ) (third_segment_speed : ℕ) : 
  total_distance = 150 ∧ total_time = 2 ∧ first_segment_speed = 50 ∧ 
  second_segment_speed = 70 → third_segment_speed = 105 := 
by 
  intros h
  sorry

end jenna_average_speed_l508_508194


namespace computer_cost_l508_508022

theorem computer_cost (C : ℝ) (h : 1.4 * C = 2240) : 
  let SP := 1.5 * C in 
  SP = 2400 := 
by
  sorry

end computer_cost_l508_508022


namespace point_on_line_segment_ratio_l508_508209

/-- Let Q be a point on line segment CD in three-dimensional space 
such that CQ:QD = 3:4. 
Then, determine the constants x and y for which Q = xC + yD, 
where C and D are points in a vector space. -/
theorem point_on_line_segment_ratio (C D Q : ℝ^3) (x y : ℝ)
  (h : ∃ t ∈ Icc(0, 1), Q = t • C + (1 - t) • D)
  (p : (∃ (m n : ℝ), m = 3 ∧ n = 4)) :
  (x, y) = (4 / 7, 3 / 7) :=
by
  sorry

end point_on_line_segment_ratio_l508_508209


namespace part1_a_neg1_part2_min_val_l508_508859

-- Assuming the coefficients as given:
noncomputable def quad_func (a x : ℝ) : ℝ := 2 * x^2 - (a + 2) * x + a

theorem part1_a_neg1 (x : ℝ) : quad_func (-1) x > 0 ↔ (x > 1 ∨ x < -1 / 2) :=
  sorry

noncomputable def quad_eq (a x : ℝ) : ℝ := 2 * x^2 - (a + 3) * x + (a - 1)

theorem part2_min_val (a : ℝ) (x1 x2 : ℝ) (h1: a > 1) 
  (h2 : quad_eq a x1 = 0) (h3 : quad_eq a x2 = 0) (hx1 : 0 < x1) (hx2 : 0 < x2) 
  (hx1x2 : x1 * x2 = (a - 1) / 2) (h_sum : x1 + x2 = (a + 3) / 2) :
  minValue (x1 x2 : ℝ) (hx1 hx2 : 0 < x1 ∧ 0 < x2) = 6 :=
  sorry

end part1_a_neg1_part2_min_val_l508_508859


namespace percentage_reduction_l508_508713

theorem percentage_reduction :
  let original := 243.75
  let reduced := 195
  let percentage := ((original - reduced) / original) * 100
  percentage = 20 :=
by
  sorry

end percentage_reduction_l508_508713


namespace jenny_popcorn_kernels_l508_508952

theorem jenny_popcorn_kernels
  (d : ℕ) (k : ℕ) (t : ℕ) (eaten_ratio : ℚ)
  (kernel_drop_rate : k = 1)
  (distance_to_school : d = 5000)
  (drop_rate : ∀ distance, distance / 25 = kernel_drop_rate * distance / 25)
  (squirrel_eats : eaten_ratio = 1 / 4)
  : t = (d / 25) - ((d / 25) * eaten_ratio) :=
by
  sorry

end jenny_popcorn_kernels_l508_508952


namespace percentage_of_identical_last_two_digits_l508_508891

theorem percentage_of_identical_last_two_digits : 
  let n := 9000 in
  let valid_numbers := 720 in
  let percentage := (valid_numbers : ℝ) * 100 / n in
  percentage = 8.0 :=
by {
  sorry
}

end percentage_of_identical_last_two_digits_l508_508891


namespace initial_seashells_l508_508551

-- Definitions for the conditions
def seashells_given_to_Tim : ℕ := 13
def seashells_now : ℕ := 36

-- Proving the number of initially found seashells
theorem initial_seashells : seashells_now + seashells_given_to_Tim = 49 :=
by
  -- we omit the proof steps with sorry
  sorry

end initial_seashells_l508_508551


namespace fraction_simplification_l508_508322

theorem fraction_simplification :
  (1^2 + 1) * (2^2 + 1) * (3^2 + 1) / ((2^2 - 1) * (3^2 - 1) * (4^2 - 1)) = 5 / 18 :=
by
  sorry

end fraction_simplification_l508_508322


namespace distinct_digit_placements_l508_508158

theorem distinct_digit_placements :
  let grid_size := 3
  let num_digits := 4
  let corner_boxes := { (1, 1), (1, 3), (3, 1), (3, 3) }
  in (∃ f : (fin grid_size × fin grid_size) → option ℕ,
        (∀ p, f p = none ∨ (f p = some 1 ∨ f p = some 2 ∨ f p = some 3 ∨ f p = some 4)) ∧
        ∃ p, p ∈ corner_boxes ∧ f p = some 1 ∧
        (∀ p₁ p₂, f p₁ = f p₂ → p₁ = p₂) ∧  -- at most one digit in each box
        (∃ p₁ p₂ p₃, f p₁ = some 2 ∧ f p₂ = some 3 ∧ f p₃ = some 4))
    → 1344
:= by
  sorry

end distinct_digit_placements_l508_508158


namespace number_of_fish_in_pond_l508_508009

theorem number_of_fish_in_pond (N : ℕ) (h1 : 120 > 0) (h2 : 100 > 0) (h3 : 10 > 0)
  (proportion_initial : 120 / N) (proportion_second_catch : 10 / 100) :
  N = 1200 := by
  sorry

end number_of_fish_in_pond_l508_508009


namespace incorrect_statement_l508_508108

-- Defining the conditions
variables {a b : ℝ} 
variables {p : ℝ} (hp : p > 0)
variables {ci : ℕ → ℝ}
variables h_geometric : ∃ r : ℝ, (r ≠ 1) ∧ (∀ n : ℕ, ci (n + 1) = r * ci n)
variables h_intersects_line : ∀ n : ℕ, ∃ x y : ℝ, (a * x + b * y + ci n = 0) ∧ (y^2 = 2 * p * x)

-- Sequence of midpoints
variables {xi yi : ℕ → ℝ} 
variables h_midpoint : ∀ n : ℕ, xi n = (some midpoint calculation based on ci, a, b, and p)
variables yi_constant : ∀ n : ℕ, yi n = 0

-- Incorrect statement proof
theorem incorrect_statement : ¬ (∃ d : ℝ, ∀ n : ℕ, xi (n + 1) = xi n + d) := 
sorry

end incorrect_statement_l508_508108


namespace sin_value_of_alpha_l508_508470

theorem sin_value_of_alpha (α : ℝ) (h1 : cos (π + α) = 1 / 3) (h2 : π < α ∧ α < 2 * π) : 
  sin α = - (2 * real.sqrt 2) / 3 :=
sorry

end sin_value_of_alpha_l508_508470


namespace convex_quadrilateral_angle_bound_l508_508227

theorem convex_quadrilateral_angle_bound (A B C D : Type) [convex_quadrilateral A B C D] :
  ∃ (α : ℝ), (α = ∠BAC ∨ α = ∠DBC ∨ α = ∠ACD ∨ α = ∠BDA) ∧ α ≤ π / 4 :=
begin
  sorry
end

end convex_quadrilateral_angle_bound_l508_508227


namespace probability_fss_condition_l508_508154

theorem probability_fss_condition (total_silverware : ℕ) (forks : ℕ) (spoons : ℕ) (knives : ℕ) (selection : ℕ) :
  (total_silverware = 20) ∧ (forks = 8) ∧ (spoons = 5) ∧ (knives = 7) ∧ (selection = 4) →
  (nat.choose total_silverware selection ≠ 0) →
  let favorable_outcomes := (nat.choose forks 2) * (nat.choose spoons 1) * (nat.choose knives 1) in
  let total_outcomes := nat.choose total_silverware selection in
  (favorable_outcomes.to_rat / total_outcomes.to_rat) = (196 : ℚ) / 969 :=
begin
  sorry -- proof not required as per procedure
end

end probability_fss_condition_l508_508154


namespace pascals_triangle_exclude_prime_rows_l508_508878

open Nat

theorem pascals_triangle_exclude_prime_rows :
  let non_prime_rows := [0, 1, 4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22, 24, 25, 26, 27, 28]
  let count_elements := fun n => n + 1
  Σ i in non_prime_rows, count_elements i = 304 :=
begin
  let non_prime_rows := [0, 1, 4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22, 24, 25, 26, 27, 28],
  let count_elements := λ n, n + 1,
  have sum_non_prime : Σ i in non_prime_rows, count_elements i = 304 :=
    by {
      sorry
    },
  exact sum_non_prime,
end

end pascals_triangle_exclude_prime_rows_l508_508878


namespace computation_correct_l508_508217

def f (x : ℝ) := x - 4
def g (x : ℝ) := x / 3
def finv (y : ℝ) := y + 4
def ginv (y : ℝ) := 3 * y

theorem computation_correct : 
  f (ginv (finv (finv (g (f 33))))) = 49 := 
  sorry

end computation_correct_l508_508217


namespace parallelogram_with_right_angle_is_rectangle_l508_508695

theorem parallelogram_with_right_angle_is_rectangle {P : Type} [parallelogram P] :
  (∃ (a b c d : P), parallelogram a b c d ∧ (angle a == 90 ∨ angle b == 90 ∨ angle c == 90 ∨ angle d == 90)) →
  (∃ (a b c d : P), rectangle a b c d) :=
by
  sorry

end parallelogram_with_right_angle_is_rectangle_l508_508695


namespace round_2_7982_to_0_01_l508_508732

theorem round_2_7982_to_0_01 : round_to (2.7982) (0.01) = 2.80 :=
by
  sorry

end round_2_7982_to_0_01_l508_508732


namespace part_two_l508_508129

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  x^2 - 2 * x + a * Real.log x

theorem part_two (a : ℝ) (h : a = 4) (m n : ℝ) (hm : 0 < m) (hn : 0 < n)
  (h_cond : (f m a + f n a) / (m^2 * n^2) = 1) : m + n ≥ 3 :=
sorry

end part_two_l508_508129


namespace tangerines_left_proof_l508_508195

-- Define the number of tangerines Jimin ate
def tangerinesJiminAte : ℕ := 7

-- Define the total number of tangerines
def totalTangerines : ℕ := 12

-- Define the number of tangerines left
def tangerinesLeft : ℕ := totalTangerines - tangerinesJiminAte

-- Theorem stating the number of tangerines left equals 5
theorem tangerines_left_proof : tangerinesLeft = 5 := 
by
  sorry

end tangerines_left_proof_l508_508195


namespace overlapping_area_zero_l508_508308

def radius_A := 3
def radius_B := 6
def distance_AB := 10

def area_of_overlapping_region (rA rB dAB : ℝ) : ℝ :=
  if dAB < rA + rB then π * ((rA + rB - dAB) / 2) ^ 2 else 0

theorem overlapping_area_zero : 
  area_of_overlapping_region radius_A radius_B distance_AB = 0 :=
by
  simp [radius_A, radius_B, distance_AB, area_of_overlapping_region]
  sorry

end overlapping_area_zero_l508_508308


namespace distance_AF_l508_508465

-- define the focus of the parabola y^2 = 4x
structure Point :=
  (x : ℝ)
  (y : ℝ)

def focus : Point := ⟨1, 0⟩

def on_parabola (A : Point) : Prop :=
  A.y ^ 2 = 4 * A.x

def midpoint_abscissa (A : Point) (F : Point) : ℝ :=
  (A.x + F.x) / 2

theorem distance_AF
  (A : Point)
  (hA : on_parabola A)
  (h_midpoint : midpoint_abscissa A focus = 2) :
  dist (A.x, A.y) (focus.x, focus.y) = 4 :=
sorry

end distance_AF_l508_508465


namespace distance_from_center_to_point_l508_508392

-- Define the circle equation condition.
def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 = 4 * x + 6 * y + 3

-- Define the center of the circle obtained from completing the square process.
def circle_center : ℝ × ℝ :=
  (2, 3)

-- Define the given point.
def given_point : ℝ × ℝ :=
  (10, -2)

-- Define the distance formula between two points.
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Prove that the distance between the circle's center and the given point is √89.
theorem distance_from_center_to_point :
  distance circle_center given_point = real.sqrt 89 :=
by
  sorry

end distance_from_center_to_point_l508_508392


namespace compute_g_gg_g2_l508_508218

def g (x : ℝ) : ℝ := x^2 - x - 1

theorem compute_g_gg_g2 : g(g(g(2))) = 1 := 
by
  sorry

end compute_g_gg_g2_l508_508218


namespace hexagon_area_ratio_l508_508606

theorem hexagon_area_ratio
  (K : ℝ) -- Assume the area of each small isosceles right triangle is K
  (h1 : is_regular_hexagon ABCDEF)
  (h2 : divides_into_isosceles_right_triangles ABCDEF)
  (h3 : area_triangle ABG = 3 * K)
  (h4 : area_triangle ACE = 9 * K) :
  (area_triangle ABG / area_triangle ACE) = 1 / 3 :=
by
  sorry

end hexagon_area_ratio_l508_508606


namespace fraction_to_decimal_l508_508072

theorem fraction_to_decimal :
  (11:ℚ) / 16 = 0.6875 :=
by
  sorry

end fraction_to_decimal_l508_508072


namespace meetings_percentage_l508_508237

def total_minutes_in_day (hours: ℕ): ℕ := hours * 60
def first_meeting_duration: ℕ := 60
def second_meeting_duration (first_meeting_duration: ℕ): ℕ := 3 * first_meeting_duration
def total_meeting_duration (first_meeting_duration: ℕ) (second_meeting_duration: ℕ): ℕ := first_meeting_duration + second_meeting_duration
def percentage_of_workday_spent_in_meetings (total_meeting_duration: ℕ) (total_minutes_in_day: ℕ): ℚ := (total_meeting_duration / total_minutes_in_day) * 100

theorem meetings_percentage (hours: ℕ) (first_meeting_duration: ℕ) (second_meeting_duration: ℕ) (total_meeting_duration: ℕ) (total_minutes_in_day: ℕ) 
(h1: total_minutes_in_day = 600) 
(h2: first_meeting_duration = 60) 
(h3: second_meeting_duration = 180) 
(h4: total_meeting_duration = 240):
percentage_of_workday_spent_in_meetings total_meeting_duration total_minutes_in_day = 40 := by
  sorry

end meetings_percentage_l508_508237


namespace factorization_correct_l508_508663

theorem factorization_correct (x : ℝ) : 
  x^4 - 5*x^2 - 36 = (x^2 + 4)*(x + 3)*(x - 3) :=
sorry

end factorization_correct_l508_508663


namespace integer_points_in_enclosed_area_l508_508632

theorem integer_points_in_enclosed_area :
  let f1 := fun x : ℝ => |x^2 - x - 2|
      f2 := fun x : ℝ => |x^2 - x|
  ∃ (S : Set (ℤ × ℤ)), S = {p : ℤ × ℤ | (f1 p.1) ≥ p.2 ∧ (f2 p.1) ≤ p.2} ∧ S.card = 6 :=
by
  let f1 := fun x : ℝ => |x^2 - x - 2|
  let f2 := fun x : ℝ => |x^2 - x|
  sorry

end integer_points_in_enclosed_area_l508_508632


namespace Natascha_distance_ratio_l508_508596

theorem Natascha_distance_ratio (r : ℝ) :
  let speed_run := r,
      speed_cycle := 3 * r,
      time_run := 1,
      time_cycle := 4,
      distance_run := time_run * speed_run,
      distance_cycle := time_cycle * speed_cycle in
  distance_cycle / distance_run = 12 :=
by
  let speed_run := r
  let speed_cycle := 3 * r
  let time_run := 1
  let time_cycle := 4
  let distance_run := time_run * speed_run
  let distance_cycle := time_cycle * speed_cycle
  sorry

end Natascha_distance_ratio_l508_508596


namespace number_of_different_words_of_length_13_is_57_l508_508765

open Real

namespace SequenceFirstDigit

-- Definition of the sequence where the nth term is the first digit of 2^n
def sequence (n : ℕ) : ℕ :=
  let log2 := log 2;
  let a_n := fractional ((n : ℝ) * log2);
  if a_n < log2 then 1 else
  if a_n < log 3 then 2 else
  if a_n < log 4 then 3 else
  if a_n < log 5 then 4 else
  if a_n < log 6 then 5 else
  if a_n < log 7 then 6 else
  if a_n < log 8 then 7 else
  if a_n < log 9 then 8 else 9

theorem number_of_different_words_of_length_13_is_57 :
  ∃ S : set (vector ℕ 13), S.card = 57 ∧ ∀ v ∈ S, ∃ n : ℕ, v = vector.of_fn (λ i, sequence (n + i)) :=
sorry

end SequenceFirstDigit

end number_of_different_words_of_length_13_is_57_l508_508765


namespace min_value_geometric_seq_sum_l508_508478

noncomputable def geometric_sequence_sum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r ^ n) / (1 - r)

theorem min_value_geometric_seq_sum :
  ∃ n : ℕ, 
  (∃ a : ℝ, 
  (∃ r : ℝ, 
  (a * r = 2 ∧ a * r ^ 4 = 16 ∧ 
    (let S_n := geometric_sequence_sum a r n in
     let S_2n := geometric_sequence_sum a r (2 * n) in
     let expression := (S_2n + S_n + 18) / (2 ^ n) in
     expression = 9)))) :=
begin
  sorry
end

end min_value_geometric_seq_sum_l508_508478


namespace max_area_of_inscribed_rectangle_l508_508766

noncomputable def maximum_area_rectangle_in_triangle (x h θ : ℝ) : ℝ :=
  (2 * x * h / (Real.tan (θ / 2))) - 2 * x^2 * (Real.cot (θ / 2))

theorem max_area_of_inscribed_rectangle
  (x h θ : ℝ)
  (isosceles_triangle : true)
  (vertex_angle : true)
  (triangle_height : true)
  (rectangle_inscribed : true)
  (rectangle_altitude : true) :
  maximum_area_rectangle_in_triangle x h θ = (2 * x * h / (Real.tan (θ / 2))) - 2 * x^2 * (Real.cot (θ / 2)) :=
by
  sorry

end max_area_of_inscribed_rectangle_l508_508766


namespace required_bike_speed_l508_508403

theorem required_bike_speed (swim_distance run_distance bike_distance swim_speed run_speed total_time : ℝ)
  (h_swim_dist : swim_distance = 0.5)
  (h_run_dist : run_distance = 4)
  (h_bike_dist : bike_distance = 12)
  (h_swim_speed : swim_speed = 1)
  (h_run_speed : run_speed = 8)
  (h_total_time : total_time = 1.5) :
  (bike_distance / ((total_time - (swim_distance / swim_speed + run_distance / run_speed)))) = 24 :=
by
  sorry

end required_bike_speed_l508_508403


namespace money_left_for_lunch_and_snacks_l508_508245

-- Definitions according to the conditions
def ticket_cost_per_person : ℝ := 5
def bus_fare_one_way_per_person : ℝ := 1.50
def total_budget : ℝ := 40
def number_of_people : ℝ := 2

-- The proposition to be proved
theorem money_left_for_lunch_and_snacks : 
  let total_zoo_cost := ticket_cost_per_person * number_of_people
  let total_bus_fare := bus_fare_one_way_per_person * number_of_people * 2
  let total_expense := total_zoo_cost + total_bus_fare
  total_budget - total_expense = 24 :=
by
  sorry

end money_left_for_lunch_and_snacks_l508_508245


namespace total_candies_is_90_l508_508599

-- Defining the conditions
def boxes_chocolate := 6
def boxes_caramel := 4
def pieces_per_box := 9

-- Defining the total number of boxes
def total_boxes := boxes_chocolate + boxes_caramel

-- Defining the total number of candies
def total_candies := total_boxes * pieces_per_box

-- Theorem stating the proof problem
theorem total_candies_is_90 : total_candies = 90 := by
  -- Provide a placeholder for the proof
  sorry

end total_candies_is_90_l508_508599


namespace road_network_transfers_l508_508672

-- Defining the types and variables
universe u
variable {City : Type u} (G : City → City → Prop)
variable [Fintype City] [DecidableRel G]

-- Declaring the main theorem based on problem statement
theorem road_network_transfers (h1 : Fintype.card City = 1993)
  (h2 : ∀ c : City, (Fintype.card (Subtype (G c))) ≥ 93)
  (h3 : ∀ a b : City, ∃ t : List City, List.chain' G t ∧ t.head = some a ∧ t.last = some b) :
  ∃ k ≤ 62, ∀ a b : City, ∃ t : List City, List.chain' G t ∧ t.head = some a ∧ t.length - 1 ≤ k ∧ t.last = some b := 
sorry

end road_network_transfers_l508_508672


namespace prime_factor_problem_l508_508647

open Nat

noncomputable def number_of_prime_factors (n : ℕ) : ℕ := 
  if n = 0 then 0 else (factors n).length

theorem prime_factor_problem (x y : ℕ) (hx : x > 0) (hy : y > 0)
  (h1 : log 10 x + 2 * log 10 (gcd x y) = 60)
  (h2 : log 10 y + 2 * log 10 (lcm x y) = 570) :
  3 * number_of_prime_factors x + 2 * number_of_prime_factors y = 880 :=
by
  sorry

end prime_factor_problem_l508_508647


namespace money_left_for_lunch_and_snacks_l508_508244

-- Definitions according to the conditions
def ticket_cost_per_person : ℝ := 5
def bus_fare_one_way_per_person : ℝ := 1.50
def total_budget : ℝ := 40
def number_of_people : ℝ := 2

-- The proposition to be proved
theorem money_left_for_lunch_and_snacks : 
  let total_zoo_cost := ticket_cost_per_person * number_of_people
  let total_bus_fare := bus_fare_one_way_per_person * number_of_people * 2
  let total_expense := total_zoo_cost + total_bus_fare
  total_budget - total_expense = 24 :=
by
  sorry

end money_left_for_lunch_and_snacks_l508_508244


namespace boat_return_time_l508_508944

theorem boat_return_time 
  (current_speed_middle : ℝ) (current_speed_bank : ℝ)
  (downstream_time : ℝ) (downstream_distance : ℝ)
  (still_water_speed := (downstream_distance / downstream_time) - current_speed_middle)
  (upstream_distance := downstream_distance)
  (effective_upstream_speed := still_water_speed - current_speed_bank) :
  effective_upstream_speed = 18 → upstream_distance / effective_upstream_speed = 20 :=
by
  intros heff
  calc 
    upstream_distance / effective_upstream_speed
        = 360 / 18 : by simp [heff, upstream_distance, effective_upstream_speed]
    ... = 20 : by norm_num

end boat_return_time_l508_508944


namespace area_of_PQRS_l508_508714

theorem area_of_PQRS 
  (ABCD_circle_radius : ℝ)
  (ABCD_area : ℝ)
  (h_ABCD_radius : ABCD_circle_radius = real.sqrt 2)
  (h_ABCD_area : ABCD_area = 4)
  (PQ_on_AD_circle : ℝ)
  (h_PQ_on_AD_circle : PQ_on_AD_circle = (real.sqrt (3) - 1) / 2)
  (PQRS_area : ℝ) : 
  PQRS_area = (2 - real.sqrt 3) :=
sorry

end area_of_PQRS_l508_508714


namespace parabola_vertex_to_standard_l508_508786

-- The conditions derived from the problem
variables {a b c : ℝ}
def vertex_form (a : ℝ) := λ x : ℝ, a * (x - 4)^2 + 4
def standard_form (a b c : ℝ) := λ x : ℝ, a * x^2 + b * x + c

theorem parabola_vertex_to_standard (h1 : vertex_form a 3 = 0)
    (h2 : standard_form (-4) b c = vertex_form (-4)) :
  a + b + c = -32 :=
by
  sorry

end parabola_vertex_to_standard_l508_508786


namespace divide_segment_l508_508778

-- Define the given line segment AB
variable (A B : Point)

-- Define the number of parts n
variable (n : ℕ)
-- A condition to ensure n is positive
variable (h_pos : 0 < n)

-- Statement to prove
theorem divide_segment (A B : Point) (n : ℕ) (h_pos : 0 < n) :
  ∃ (P : Fin n → Point) (h : ∀ i : Fin (n-1), (distance P i P (i+1) = distance P 0 P 1)), 
  True := 
sorry

end divide_segment_l508_508778


namespace find_m_l508_508869

-- Given definitions and conditions
def vector1 : ℝ × ℝ := (-2, 3)
def vector2 (m : ℝ) : ℝ × ℝ := (3, m)

-- The condition - vectors are perpendicular
def is_perpendicular (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

-- Final proof statement
theorem find_m : ∃ m : ℝ, is_perpendicular vector1 (vector2 m) ∧ m = 2 :=
by
  unfold is_perpendicular
  simp
  use 2
  split
  {
    norm_num,
  }
  {
    refl,
  }

end find_m_l508_508869


namespace part1_part2_l508_508864

def sequence_a : Nat → ℝ
| 0       => 1
| (n + 1) => sequence_a n + 1 / (sequence_a n)^2

theorem part1 (n : ℕ) (h : n ≥ 2) : (sequence_a n)^3 > 3 * n :=
sorry

theorem part2 (n : ℕ) (h : n ≥ 4) : ⌊sequence_a (9 * n^3)⌋ = 3 * n :=
sorry

end part1_part2_l508_508864


namespace tangents_quad_sums_eq_l508_508651

def Circle : Type := ℝ × ℝ × ℝ  -- Representing a circle by its center (x, y) and radius

structure TangentPoints :=
  (O A D B C : ℝ × ℝ)  -- Points of tangency and the tangency point of the circles

noncomputable def lengths_of_equal_tangents (P Q : ℝ × ℝ) : Prop :=
  dist P Q = dist Q P

noncomputable def sums_of_opposite_sides_equal (tp : TangentPoints) : Prop :=
  let ⟨O, A, D, B, C⟩ := tp in
  dist A B + dist C D = dist A D + dist B C

theorem tangents_quad_sums_eq (tp : TangentPoints) 
  (h1 : lengths_of_equal_tangents tp.A tp.D)
  (h2 : lengths_of_equal_tangents tp.B tp.C) : 
  sums_of_opposite_sides_equal tp :=
begin
  sorry
end

end tangents_quad_sums_eq_l508_508651


namespace hyperbola_eccentricity_l508_508163

-- Definitions based on conditions
def hyperbola_eqn (a : ℝ) : Prop := ∀ x y : ℝ, (x^2) / a - (y^2) / 4 = 1
def asymptote (a : ℝ) (x y : ℝ) : Prop := x - 2 * y = 0

-- Statement of the problem
theorem hyperbola_eccentricity (a : ℝ) (h_eqn : hyperbola_eqn a) (h_asym : asymptote a) : 
  (e : ℝ) (h_eccentricity : e = √5 / 2) : e = √5 / 2 :=
by
  sorry

end hyperbola_eccentricity_l508_508163


namespace similarity_of_triangles_circumcircles_pass_through_D_l508_508645

-- Given conditions
variables {A B1 B2 C1 C2 D : Type}
variables [Euclidean_geometry] -- Assuming Euclidean geometry context
variable [is_not_same_point: A ≠ D]
variables [similar_triangles: geometric_similar_triangles AB1C1 ABC2C2]
variables [same_orientation: same_orientation AB1C1 ABC2C2]
variables [intersect_B1B2_C1C2: line_intersection B1B2 C1C2 D]

-- Part (a): Prove that the triangles AB1B2 and AC1C2 are similar
theorem similarity_of_triangles (h: A ≠ D):
  similar_triangles AB1B2 AC1C2 :=
sorry

-- Part (b): Prove that the circumcircles pass through point D
theorem circumcircles_pass_through_D (h: A ≠ D):
  pass_through_circumcircles AB1C1 ABC2C2 D :=
sorry

end similarity_of_triangles_circumcircles_pass_through_D_l508_508645


namespace invertible_functions_l508_508051

noncomputable theory
open Classical

def p (x : ℝ) : ℝ := real.sqrt (3 - x)
def q (x : ℝ) : ℝ := x^3 - 3 * x
def r (x : ℝ) : ℝ := x + 2 / x
def s (x : ℝ) : ℝ := 3 * x^2 + 6 * x + 8
def t (x : ℝ) : ℝ := abs (x - 1) + abs (x + 2)
def u (x : ℝ) : ℝ := 2^x + 8^x
def v (x : ℝ) : ℝ := x - 2 / x
def w (x : ℝ) : ℝ := x / 3

lemma p_invertible : ∃ f : ℝ → ℝ, ∀ y ∈ set.Iic 3, p (f y) = y ∧ f (p y) = y := by sorry
lemma s_invertible : ∃ f : ℝ → ℝ, ∀ y ∈ set.Ici 1, s (f y) = y ∧ f (s y) = y := by sorry
lemma u_invertible : ∃ f : ℝ → ℝ, ∀ y : ℝ, u (f y) = y ∧ f (u y) = y := by sorry
lemma v_invertible : ∃ f : ℝ → ℝ, ∀ y ∈ set.Ioi 0, v (f y) = y ∧ f (v y) = y := by sorry
lemma w_invertible : ∃ f : ℝ → ℝ, ∀ y ∈ set.Ico (-3) 9, w (f y) = y ∧ f (w y) = y := by sorry

theorem invertible_functions :
  (∃ f : ℝ → ℝ, ∀ y ∈ set.Iic 3, p (f y) = y ∧ f (p y) = y) ∧
  (∃ f : ℝ → ℝ, ∀ y ∈ set.Ici 1, s (f y) = y ∧ f (s y) = y) ∧
  (∃ f : ℝ → ℝ, ∀ y : ℝ, u (f y) = y ∧ f (u y) = y) ∧
  (∃ f : ℝ → ℝ, ∀ y ∈ set.Ioi 0, v (f y) = y ∧ f (v y) = y) ∧
  (∃ f : ℝ → ℝ, ∀ y ∈ set.Ico (-3) 9, w (f y) = y ∧ f (w y) = y) := by
  exact ⟨p_invertible, s_invertible, u_invertible, v_invertible, w_invertible⟩

end invertible_functions_l508_508051


namespace determinant_equality_l508_508204

noncomputable def det_matrix (A B : Real) : Real :=
  Matrix.det ![
    ![Real.cot A, 1, 0.5],
    ![0.5, Real.cot B, 1],
    ![1, 0.5, Real.cot (Real.pi / 2 - A)]
  ]

theorem determinant_equality (A B : Real) (h1 : A + B = Real.pi / 2) : 
  det_matrix A B = 5 / 8 := 
  sorry

end determinant_equality_l508_508204


namespace sqrt_diff_is_neg_ten_l508_508622

def sqrt_diff_integer : ℤ :=
  sqrt (abs (40 * real.sqrt 2 - 57)) - sqrt (40 * real.sqrt 2 + 57)

theorem sqrt_diff_is_neg_ten : sqrt_diff_integer = -10 :=
  sorry

end sqrt_diff_is_neg_ten_l508_508622


namespace smallest_b_for_quadratic_factorization_l508_508799

theorem smallest_b_for_quadratic_factorization : ∃ (b : ℕ), 
  (∀ r s : ℤ, (r * s = 4032) ∧ (r + s = b) → b ≥ 127) ∧ 
  (∃ r s : ℤ, (r * s = 4032) ∧ (r + s = b) ∧ (b = 127))
:= sorry

end smallest_b_for_quadratic_factorization_l508_508799


namespace min_even_integers_among_eight_l508_508649

theorem min_even_integers_among_eight :
  ∃ (x y z a b m n o : ℤ), 
    x + y + z = 30 ∧
    x + y + z + a + b = 49 ∧
    x + y + z + a + b + m + n + o = 78 ∧
    (∀ e : ℕ, (∀ x y z a b m n o : ℤ, x + y + z = 30 ∧ x + y + z + a + b = 49 ∧ x + y + z + a + b + m + n + o = 78 → 
    e = 2)) := sorry

end min_even_integers_among_eight_l508_508649


namespace solve_for_x_l508_508323

theorem solve_for_x (x : ℝ) (h : x ≠ 0) : (3 * x)^5 = (9 * x)^4 → x = 27 := 
by 
  admit

end solve_for_x_l508_508323


namespace maximum_value_expression_l508_508220

-- Defining the variables and the main condition
variables (x y z : ℝ)

-- Assuming the non-negativity and sum of squares conditions
variables (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (hxyz : x^2 + y^2 + z^2 = 1)

-- Main statement about the maximum value
theorem maximum_value_expression : 
  4 * x * y * Real.sqrt 2 + 5 * y * z + 3 * x * z * Real.sqrt 3 ≤ 
  (44 * Real.sqrt 2 + 110 + 9 * Real.sqrt 3) / 3 :=
sorry

end maximum_value_expression_l508_508220


namespace point_in_second_quadrant_l508_508841

theorem point_in_second_quadrant (z : ℂ) (h : z / (1 + complex.I) = 2 * complex.I) : 
  z = -2 + 2 * complex.I :=
by {
  sorry
}

end point_in_second_quadrant_l508_508841


namespace max_y_coordinate_on_graph_of_cos_2theta_l508_508795

theorem max_y_coordinate_on_graph_of_cos_2theta :
  (∃ θ, (θ ∈ Set.Icc (-Real.pi) Real.pi) ∧ 
        ∀ θ', θ' ∈ Set.Icc (-Real.pi) Real.pi → 
        let y := Real.cos (2 * θ') * Real.sin θ' in 
          y ≤ Real.cos (2 * θ) * Real.sin θ) → 
  ∃ θ, θ ∈ Set.Icc (-Real.pi) Real.pi ∧ Real.cos (2 * θ) * Real.sin θ = (2 * Real.sqrt 6) / 3 := 
sorry

end max_y_coordinate_on_graph_of_cos_2theta_l508_508795


namespace scientific_notation_of_11580000_l508_508923

theorem scientific_notation_of_11580000 :
  11_580_000 = 1.158 * 10^7 :=
sorry

end scientific_notation_of_11580000_l508_508923


namespace valid_parentheses_eq_catalan_l508_508023

-- Define the Catalan number
def catalan : ℕ → ℕ
| 0 := 1
| (n + 1) := (∑ k in Finset.range (n + 1), catalan k * catalan (n - k))

-- Define the number of valid parenthetical arrangements
def number_of_parenthetical_arrangements : ℕ → ℕ := sorry

-- Prove that the number of valid parenthetical arrangements is equal to the nth Catalan number
theorem valid_parentheses_eq_catalan (n : ℕ) : number_of_parenthetical_arrangements (n + 1) = catalan n :=
sorry

end valid_parentheses_eq_catalan_l508_508023


namespace product_of_c_l508_508085

theorem product_of_c (f : ℕ → ℕ) (c : ℕ) (h1 : f 0 = 1)
  (h2 : ∀ n, f (n + 1) = (n + 1) * f n) 
  (h3 : ∀ c, (∑ k in finset.Icc 1 19, k) = 19) : 
  (∏ k in finset.Icc 1 19, k) = 121645100408832000 :=
by
  sorry

end product_of_c_l508_508085


namespace max_value_f1_solve_inequality_f2_l508_508858

def f_1 (x : ℝ) : ℝ := |x + 1| - |x - 1|

theorem max_value_f1 : ∃ x, f_1 x = 2 :=
sorry

def f_2 (x : ℝ) : ℝ := |2 * x - 1| - |x - 1|

theorem solve_inequality_f2 (x : ℝ) : f_2 x ≥ 1 ↔ x ≤ -1 ∨ x ≥ 1 :=
sorry

end max_value_f1_solve_inequality_f2_l508_508858


namespace price_difference_l508_508377

noncomputable def originalPriceStrawberries (s : ℝ) (sale_revenue_s : ℝ) := sale_revenue_s / (0.70 * s)
noncomputable def originalPriceBlueberries (b : ℝ) (sale_revenue_b : ℝ) := sale_revenue_b / (0.80 * b)

theorem price_difference
    (s : ℝ) (sale_revenue_s : ℝ)
    (b : ℝ) (sale_revenue_b : ℝ)
    (h1 : sale_revenue_s = 70 * (0.70 * s))
    (h2 : sale_revenue_b = 50 * (0.80 * b)) :
    originalPriceStrawberries (sale_revenue_s / 49) sale_revenue_s - originalPriceBlueberries (sale_revenue_b / 40) sale_revenue_b = 0.71 :=
by
  sorry

end price_difference_l508_508377


namespace circle_radius_zero_l508_508086

-- Define the given circle equation
def circle_eq (x y : ℝ) : Prop := x^2 - 4 * x + y^2 - 6 * y + 13 = 0

-- The proof problem statement
theorem circle_radius_zero : ∀ (x y : ℝ), circle_eq x y → 0 = 0 :=
by
  sorry

end circle_radius_zero_l508_508086


namespace find_amplitude_l508_508388

noncomputable def amplitude (a b c d x : ℝ) := a * Real.sin (b * x + c) + d

theorem find_amplitude (a b c d : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) 
  (h_range : ∀ x, -1 ≤ amplitude a b c d x ∧ amplitude a b c d x ≤ 7) :
  a = 4 :=
by
  sorry

end find_amplitude_l508_508388


namespace problem_l508_508840

noncomputable def f (n : ℕ) : ℚ := ∑ i in finset.range (2 * n + 1) \ finset.range (n + 1), (1 : ℚ) / (n + 1 + i)

theorem problem (n : ℕ) : 
  (f (n + 1) - f n) = (1 / (2 * n + 1) + 1 / (2 * n + 2) - 1 / (n + 1)) :=
by
  sorry

end problem_l508_508840


namespace max_n_sum_of_squares_l508_508316

theorem max_n_sum_of_squares (n : ℕ) (s : fin n → ℕ) (h_distinct : function.injective s) :
  ∑ i, (s i) ^ 2 = 1955 → n ≤ 16 :=
by sorry

end max_n_sum_of_squares_l508_508316


namespace election_total_votes_l508_508668

theorem election_total_votes (V : ℝ)
  (h_majority : ∃ O, 0.84 * V = O + 476)
  (h_total_votes : ∀ O, V = 0.84 * V + O) :
  V = 700 :=
sorry

end election_total_votes_l508_508668


namespace max_y_coordinate_polar_curve_l508_508793

theorem max_y_coordinate_polar_curve :
  ∀ (θ : ℝ), let r := cos (2 * θ),
                 x := r * cos θ,
                 y := r * sin θ in
  ∃ (θ_max : ℝ), y = (3 * real.sqrt 3) / 4 := 
by
  sorry -- Proof is omitted

end max_y_coordinate_polar_curve_l508_508793


namespace mother_gave_80_cents_l508_508235

theorem mother_gave_80_cents (father_uncles_gift : Nat) (spent_on_candy current_amount : Nat) (gift_from_father gift_from_uncle add_gift_from_uncle : Nat) (x : Nat) :
  father_uncles_gift = gift_from_father + gift_from_uncle ∧
  father_uncles_gift = 110 ∧
  spent_on_candy = 50 ∧
  current_amount = 140 ∧
  gift_from_father = 40 ∧
  gift_from_uncle = 70 ∧
  add_gift_from_uncle = 70 ∧
  x = current_amount + spent_on_candy - father_uncles_gift ∧
  x = 190 - 110 ∨
  x = 80 :=
  sorry

end mother_gave_80_cents_l508_508235


namespace find_x_in_terms_of_N_l508_508773

theorem find_x_in_terms_of_N (N : ℤ) (x y : ℝ) 
(h1 : (⌊x⌋ : ℤ) + 2 * y = N + 2) 
(h2 : (⌊y⌋ : ℤ) + 2 * x = 3 - N) : 
x = (3 / 2) - N := 
by
  sorry

end find_x_in_terms_of_N_l508_508773


namespace exists_root_in_interval_l508_508775

noncomputable def f (x : ℝ) : ℝ := 2^x - x^3

theorem exists_root_in_interval :
  ∃ c ∈ set.Ioo (1 : ℝ) 2, f c = 0 :=
begin
  have h_cont : continuous f := sorry, -- Assume or prove f is continuous.
  have h1 : f 1 > 0 := by norm_num, -- f(1) = 2^1 - 1^3 = 2 - 1 = 1 > 0
  have h2 : f 2 < 0 := by norm_num, -- f(2) = 2^2 - 2^3 = 4 - 8 = -4 < 0
  exact intermediate_value_interval h_cont h1 h2 sorry,
end

end exists_root_in_interval_l508_508775


namespace positive_cell_with_row_sum_larger_than_column_sum_l508_508536

-- Defining m, n, and the matrix A with the given properties
variables (m n : ℕ) (A : ℕ → ℕ → ℝ)
hypothesis hmn : m < n
hypothesis hA_nonneg : ∀ i j, 0 ≤ A i j
hypothesis hA_col_pos : ∀ j, ∃ i, 0 < A i j

-- Statement of the theorem
theorem positive_cell_with_row_sum_larger_than_column_sum :
  ∃ i j, 0 < A i j ∧ (∑ k in finset.range n, A i k) > (∑ k in finset.range m, A k j) :=
sorry  -- This is a placeholder for the proof.

end positive_cell_with_row_sum_larger_than_column_sum_l508_508536


namespace find_1000th_non_square_non_cube_theorem_1000th_non_square_non_cube_l508_508034

theorem find_1000th_non_square_non_cube :
  ∃ n : ℕ, n = 1039 ∧ (∀ k ≤ 1000, k ∈ non_square_non_cube_sequence k) := sorry

-- Defining what it means to be in the sequence of non-square, non-cube integers
def non_square_non_cube (n : ℕ) : Prop :=
  ¬(∃ m : ℕ, m^2 = n) ∧ ¬(∃ m : ℕ, m^3 = n)
  
-- The sequence function definition
noncomputable def non_square_non_cube_sequence (k : ℕ) : ℕ :=
  Classical.some (Nat.find_spec (λ n, non_square_non_cube n ∧
    (Nat.find (λ m, non_square_non_cube m ∧ (n = m)) = k)))

-- The main theorem to find the 1000th number in the sequence
theorem theorem_1000th_non_square_non_cube :
  non_square_non_cube_sequence 1000 = 1039 := sorry

end find_1000th_non_square_non_cube_theorem_1000th_non_square_non_cube_l508_508034


namespace max_balloons_with_number_2_l508_508223

/-- Define the set S with 100 elements from natural numbers not exceeding 10,000 -/
variable (S : Set ℕ) [Finite S] (hS : S.card = 100 ∧ ∀ s ∈ S, s ≤ 10000)

/-- Maximum number of balloons with the number 2 written on them -/
theorem max_balloons_with_number_2 (S : Set ℕ) [Finite S] (hS : S.card = 100 ∧ ∀ s ∈ S, s ≤ 10000) :
  let points := (S × S × S : Set (ℕ × ℕ × ℕ))
  3 * (S.card.choose 2) = 14850 :=
by
  sorry

end max_balloons_with_number_2_l508_508223


namespace inequality_proof_l508_508448

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 2) : 
  (1 + 1/x) * (1 + 1/y) ≥ 4 :=
sorry

end inequality_proof_l508_508448


namespace matrix_B_plus_10_B_inv_eq_12_I_l508_508115

open Matrix

noncomputable theory

variables {R : Type*} [Field R] {n : Type*} [Fintype n] [DecidableEq n]

theorem matrix_B_plus_10_B_inv_eq_12_I (B : Matrix n n R) (I : Matrix n n R)
  (h1 : B * B⁻¹ = I) 
  (h2 : (B - 3 • I) * (B - 5 • I) = 0) :
  B + 10 • B⁻¹ = 12 • I := 
sorry

end matrix_B_plus_10_B_inv_eq_12_I_l508_508115


namespace number_of_blue_marbles_l508_508349

noncomputable def probability_red_or_white : ℝ := 0.9166666666666666

theorem number_of_blue_marbles 
  (total_marbles : ℕ := 60) 
  (red_marbles : ℕ := 9) 
  (probability_rw : ℝ = probability_red_or_white) : 
  ∃ blue_marbles : ℕ, 
    let white_marbles := total_marbles - (red_marbles + blue_marbles) in
    (white_marbles + red_marbles).to_nat / total_marbles.to_nat = probability_red_or_white ∧ 
    blue_marbles = 5 :=
by 
  apply Exists.intro 5
  sorry

end number_of_blue_marbles_l508_508349


namespace triangle_de_length_l508_508545

theorem triangle_de_length (A B C D E : Type) 
  [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E]
  (BC: ℝ) (angle_C: ℝ) (AC: ℝ)
  [hBC : BC = 30] 
  [hC : angle_C = 45] 
  [hAC : AC = 45]
  (midpoint_D : ∀ (D : Type), D = (BC / 2))
  (perpendicular_bisector : ∀ (E : Type), E = ( (AC / sqrt(2)) )):
  (D, E، ∈ BC) → 
  DE = 15 / sqrt(2) :=
sorry

end triangle_de_length_l508_508545


namespace bahs_for_1000_yahs_l508_508892

-- Definitions based on given conditions
def bahs_to_rahs_ratio (b r : ℕ) := 15 * b = 24 * r
def rahs_to_yahs_ratio (r y : ℕ) := 9 * r = 15 * y

-- Main statement to prove
theorem bahs_for_1000_yahs (b r y : ℕ) (h1 : bahs_to_rahs_ratio b r) (h2 : rahs_to_yahs_ratio r y) :
  1000 * y = 375 * b :=
by
  sorry

end bahs_for_1000_yahs_l508_508892


namespace total_selection_ways_l508_508500

-- Defining the conditions
def groupA_male_students : ℕ := 5
def groupA_female_students : ℕ := 3
def groupB_male_students : ℕ := 6
def groupB_female_students : ℕ := 2

-- Define combinations (choose function)
noncomputable def C (n k : ℕ) : ℕ := Nat.choose n k

-- The required theorem statement
theorem total_selection_ways :
  C groupA_female_students 1 * C groupA_male_students 1 * C groupB_male_students 2 +
  C groupB_female_students 1 * C groupB_male_students 1 * C groupA_male_students 2 = 345 :=
by
  sorry

end total_selection_ways_l508_508500


namespace round_2_7982_to_0_01_l508_508734

theorem round_2_7982_to_0_01 : round_to (2.7982) (0.01) = 2.80 :=
by
  sorry

end round_2_7982_to_0_01_l508_508734


namespace fractional_part_sum_l508_508572

theorem fractional_part_sum {n a b : ℤ} (hn : n > 0) (hp : Nat.gcd a n = 1) :
  (Finset.sum (Finset.range n) (λ k, fract ((k * a + b) / n))) = (n - 1) / 2 :=
by
  sorry

end fractional_part_sum_l508_508572


namespace scientific_notation_of_11580000_l508_508924

theorem scientific_notation_of_11580000 :
  11_580_000 = 1.158 * 10^7 :=
sorry

end scientific_notation_of_11580000_l508_508924


namespace normal_vector_parallel_l508_508588

theorem normal_vector_parallel {k : ℝ} 
  (n1 : ℝ × ℝ × ℝ) 
  (n2 : ℝ × ℝ × ℝ) 
  (h1 : n1 = (1, 2, -2)) 
  (h2 : n2 = (-2, -4, k)) 
  (h_parallel : ∃ (c : ℝ), ∀ i : fin 3, (n2.1 = c * n1.1) ∧ (n2.2 = c * n1.2) ∧ (n2.3 = c * n1.3)) :
  k = 4 :=
begin
  sorry
end

end normal_vector_parallel_l508_508588


namespace tiles_purchased_l508_508068

-- Problem Specification
def room_length : ℕ := 12
def room_width : ℕ := 20
def tile_area : ℕ := 1
def fraction_tiled : ℚ := 1 / 6

-- Formalization Result
theorem tiles_purchased : (room_length * room_width : ℚ) * fraction_tiled = 40 := by
  have total_area : ℕ := room_length * room_width
  have tiled_area : ℚ := total_area * fraction_tiled
  have number_of_tiles : ℚ := tiled_area / tile_area
  have : total_area = 240 := by
    calc
      12 * 20 = 240 : by norm_num
  have : tiled_area = 40 := by
    calc
      240 * (1 / 6) = 40 : by norm_num
  show 240 * (1 / 6 : ℚ) = 40 from this

end tiles_purchased_l508_508068


namespace find_AY_l508_508073

-- Definitions based on conditions
def divides_ratio (AB AY BY : ℝ) := BY = (4 / 7) * AB ∧ AY = (3 / 7) * AB
def bisects_angle (AC BC AY BY : ℝ) := (AC / AY) = (BC / BY)

-- Given values
def AB := 40.0
def AC := 27.0
def BC := 35.0
def BY := (4 / 7) * AB

-- The theorem we need to prove
theorem find_AY (AY : ℝ) : divides_ratio AB AY BY ∧ bisects_angle AC BC AY BY → AY = 864 / 49 :=
by
  sorry

end find_AY_l508_508073


namespace draw_white_ball_probability_l508_508294

noncomputable def probability_of_drawing_white_ball : ℝ :=
  let urn1 := {black := 2, white := 3}
  let urn2 := {black := 2, white := 1}
  let prob_white_urn1 : ℝ := urn1.white / (urn1.white + urn1.black)
  let prob_white_urn2 : ℝ := urn2.white / (urn2.white + urn2.black)
  let prob_choose_urn1 : ℝ := 1 / 2
  let prob_choose_urn2 : ℝ := 1 / 2
  prob_white_urn1 * prob_choose_urn1 + prob_white_urn2 * prob_choose_urn2

theorem draw_white_ball_probability : probability_of_drawing_white_ball = 7 / 15 := by
  sorry

end draw_white_ball_probability_l508_508294


namespace real_root_polynomials_l508_508788

open Polynomial

theorem real_root_polynomials (n : ℕ) (P : Polynomial ℝ) :
  (∀ i, i < n → (coeff P (n - 1 - i) = 1 ∨ coeff P (n - 1 - i) = -1)) →
  (∀ r, is_root P r → r ∈ ℝ) →
  (n = 1 ∧ (P = X - 1 ∨ P = X + 1)) ∨
  (n = 2 ∧ (P = X^2 - 1 ∨ P = X^2 - X - 1 ∨ P = X^2 + X - 1)) ∨
  (n = 3 ∧ (P = (X^2 - 1) * (X - 1) ∨ P = (X^2 - 1) * (X + 1))) :=
by sorry

end real_root_polynomials_l508_508788


namespace w_identity_l508_508109

theorem w_identity (w : ℝ) (h_pos : w > 0) (h_eq : w - 1 / w = 5) : (w + 1 / w) ^ 2 = 29 := by
  sorry

end w_identity_l508_508109


namespace smallest_alpha_exists_l508_508427

-- Assuming definitions for convex polygon, central symmetry and convex hull are present in Mathlib.
structure ConvexPolygon (P : Type) :=
(area : ℝ)
(is_convex : Prop)

def central_symmetry (P : ConvexPolygon) (M : ℝ × ℝ) : ConvexPolygon :=
sorry -- Definition depending on the existing mathlib components

def convex_hull (P Q : ConvexPolygon) : ConvexPolygon :=
sorry -- Definition depending on the existing mathlib components

noncomputable def min_alpha : ℝ := 4 / 3

theorem smallest_alpha_exists :
  ∀ (P : ConvexPolygon),
  P.area = 1 →
  ∃ (M : ℝ × ℝ), (convex_hull P (central_symmetry P M)).area ≤ min_alpha :=
sorry

end smallest_alpha_exists_l508_508427


namespace interest_rate_fund_x_l508_508594

-- Variables declaration
variables (total_investment : ℕ) (fund_x : ℕ) (fund_y_rate : ℝ) (interest_diff : ℝ) 

-- Let’s state our conditions
variables (h1 : total_investment = 100000) -- Total investment in fund X and fund Y
variables (h2 : fund_x = 42000) -- Investment in fund X
variables (h3 : fund_y_rate = 0.17) -- Interest rate of fund Y
variables (h4 : interest_diff = 200) -- Interest difference between fund Y and fund X

-- Goal: To prove that the interest rate of fund X is 0.23
theorem interest_rate_fund_x (r : ℝ) : 
  let fund_y := total_investment - fund_x in
  let interest_x := fund_x * r in
  let interest_y := fund_y * fund_y_rate in
  interest_y = interest_x + interest_diff →
  r = 0.23 :=
by 
  sorry

end interest_rate_fund_x_l508_508594


namespace number_of_valid_arrangements_l508_508935

theorem number_of_valid_arrangements :
    ∃ (n : ℕ), n = 6 * 4^6 * 5^3 + 9 * 4^4 * 5^5 + 5^9 :=
by
  use 6 * 4^6 * 5^3 + 9 * 4^4 * 5^5 + 5^9
  rfl

end number_of_valid_arrangements_l508_508935


namespace max_F_of_eternal_number_l508_508903

-- Definitions of digits and the number M
def is_eternal_number (a b c d : ℕ) : Prop :=
  b + c + d = 12 ∧ a = b - d

-- Definition of the function F(M)
def F (a b c d : ℕ) : ℤ :=
  100 * a - 100 * b + c - d

-- The mathematical statement to be proved
theorem max_F_of_eternal_number (a b c d : ℕ) (h1 : is_eternal_number a b c d)
    (h2 : ∃ k : ℤ, F a b c d = 9 * k) : 
    F a b c d ≤ 9 :=
begin
  sorry
end

end max_F_of_eternal_number_l508_508903


namespace product_floor_ceil_sequence_l508_508406

theorem product_floor_ceil_sequence :
  (Int.floor (-6 - 0.5) * Int.ceil (6 + 0.5) *
   Int.floor (-5 - 0.5) * Int.ceil (5 + 0.5) *
   Int.floor (-4 - 0.5) * Int.ceil (4 + 0.5) *
   Int.floor (-3 - 0.5) * Int.ceil (3 + 0.5) *
   Int.floor (-2 - 0.5) * Int.ceil (2 + 0.5) *
   Int.floor (-1 - 0.5) * Int.ceil (1 + 0.5) *
   Int.floor (-0.5) * Int.ceil (0.5)) = -25401600 := sorry

end product_floor_ceil_sequence_l508_508406


namespace fractional_part_sum_l508_508574

noncomputable def fractional_part (x : ℚ) : ℚ := x - ⌊x⌋

theorem fractional_part_sum (n a b : ℤ) (hn : n > 0) (ha_coprime : Nat.coprime a.natAbs n.natAbs) :
  (∑ k in Finset.range n, fractional_part ((↑k * a + b) / n)) = (n - 1) / 2 :=
by
    sorry

end fractional_part_sum_l508_508574


namespace max_F_of_eternal_number_l508_508904

-- Definitions of digits and the number M
def is_eternal_number (a b c d : ℕ) : Prop :=
  b + c + d = 12 ∧ a = b - d

-- Definition of the function F(M)
def F (a b c d : ℕ) : ℤ :=
  100 * a - 100 * b + c - d

-- The mathematical statement to be proved
theorem max_F_of_eternal_number (a b c d : ℕ) (h1 : is_eternal_number a b c d)
    (h2 : ∃ k : ℤ, F a b c d = 9 * k) : 
    F a b c d ≤ 9 :=
begin
  sorry
end

end max_F_of_eternal_number_l508_508904


namespace ingrid_income_l508_508196

variables (I : ℝ)

def john_income : ℝ := 56000
def john_tax_rate : ℝ := 0.30
def ingrid_tax_rate : ℝ := 0.40
def combined_tax_rate : ℝ := 0.3569

-- John pays 30% tax on his income of 56000
def john_tax : ℝ := john_tax_rate * john_income

-- Ingrid pays 40% tax on her income I
def ingrid_tax : ℝ := ingrid_tax_rate * I

-- Their combined income
def combined_income : ℝ := john_income + I

-- Their combined tax
def combined_tax : ℝ := john_tax + ingrid_tax

-- Combined tax rate equation
def tax_rate_equation : Prop := (combined_tax / combined_income) = combined_tax_rate

-- Proving Ingrid's income
theorem ingrid_income : tax_rate_equation → I = 73924.13 :=
by
  intros,
  sorry

end ingrid_income_l508_508196


namespace divisor_condition_l508_508801

def M (n : ℤ) : Set ℤ := {n, n+1, n+2, n+3, n+4}

def S (n : ℤ) : ℤ := 5*n^2 + 20*n + 30

def P (n : ℤ) : ℤ := (n * (n+1) * (n+2) * (n+3) * (n+4))^2

theorem divisor_condition (n : ℤ) : S n ∣ P n ↔ n = 3 := 
by
  sorry

end divisor_condition_l508_508801


namespace overall_gain_is_correct_l508_508712

-- Definitions of the conditions
def CP1 : ℝ := 100
def gain1 : ℝ := 0.15
def SP1 : ℝ := CP1 * (1 + gain1)

def CP2 : ℝ := 150
def gain2 : ℝ := 0.20
def SP2 : ℝ := CP2 * (1 + gain2)

def CP3 : ℝ := 200
def loss3 : ℝ := 0.10
def SP3 : ℝ := CP3 * (1 - loss3)

def CP4 : ℝ := 250
def gain4 : ℝ := 0.12
def SP4 : ℝ := CP4 * (1 + gain4)

def total_cost_price : ℝ := CP1 + CP2 + CP3 + CP4
def total_selling_price : ℝ := SP1 + SP2 + SP3 + SP4
def overall_gain : ℝ := total_selling_price - total_cost_price

def overall_gain_percentage : ℝ := (overall_gain / total_cost_price) * 100

-- Statement to prove
theorem overall_gain_is_correct :
  overall_gain_percentage = 7.857 := by
  sorry

end overall_gain_is_correct_l508_508712


namespace largest_constant_C_exists_l508_508422

theorem largest_constant_C_exists (C : ℝ) :
  (∀ (n : ℕ) (h_pos_n : n > 0) (a b : fin n → ℕ),
    (∀ (i j : fin n), i ≠ j → (a i, b i) ≠ (a j, b j)) →
    ∑ i in fin.range n, max (a i) (b i) ≥ C * n * real.sqrt n) ↔ (C ≤ 2 * real.sqrt 2 / 3) :=
sorry

end largest_constant_C_exists_l508_508422


namespace exponent_for_base_4_l508_508893

/-- 
To find the exponent for the base 4 in the equation 16^y = 4^n 
given that y = 7, we show that the exponent n is 14.
-/

theorem exponent_for_base_4 (y : ℕ) (hyp : y = 7) : ∃ n : ℕ, 16 ^ y = 4 ^ n ∧ n = 14 :=
by
  use 14
  split
  · rw [hyp, pow_succ, pow_succ, pow_succ, pow_succ, pow_succ, pow_succ, pow_succ]
    -- Show the equality holds
    sorry
  · sorry

end exponent_for_base_4_l508_508893


namespace sum_of_roots_l508_508438

theorem sum_of_roots :
  let a := (6 : ℝ) + 3 * Real.sqrt 3
  let b := (3 : ℝ) + Real.sqrt 3
  let c := -(3 : ℝ)
  let root_sum := -b / a
  root_sum = -1 + Real.sqrt 3 / 3 := sorry

end sum_of_roots_l508_508438


namespace cars_catch_up_pedestrian_l508_508145

noncomputable def pedestrian_speed := (200 : ℕ) / 3
noncomputable def car_speed := (600 : ℕ) / 3

def pedestrian_start_time := 8 * 60 + 15  -- in minutes
def car_start_times := [8 * 60 + 20, 8 * 60 + 30, 8 * 60 + 40, 8 * 60 + 50, 9 * 60 + 0, 9 * 60 + 10, 9 * 60 + 20, 9 * 60 + 30]  -- in minutes

def time_to_catch_up (k : ℕ) : ℕ := (5 + 10 * (k - 1)) / 2  -- in minutes
def catch_up_time (start_time : ℕ) (k : ℕ) : ℕ := start_time + time_to_catch_up k 

def distance_pedestrian_cover (time : ℕ) : ℕ := (time * pedestrian_speed).to_nat
def distance_to_catch_up (k : ℕ) : ℕ := distance_pedestrian_cover ((5 + 10 * (k - 1)) / 2) + k * 1000

theorem cars_catch_up_pedestrian :
  ∀ k, k ∈ (finset.range 8).image (λ i, i + 1) → 
  ∃ time distance, 
    time = catch_up_time (car_start_times.nth (k - 1)).get_or_else 0 k ∧
    distance = distance_to_catch_up k :=
sorry

end cars_catch_up_pedestrian_l508_508145


namespace part1_part2_l508_508861

theorem part1 (m : ℝ) (h_slope : (4 - m) / (-m) = 2) : m = -4 :=
sorry

theorem part2 (m : ℝ) (h1 : 0 < m) (h2 : m < 4) :
  let S := m * (4 - m) / 2 in
  S ≤ 2 ∧ (S = 2 → m = 2 ∧ (∀ x y : ℝ, (x / 2 + y / 2 = 1) ↔ (x + y = 2))) :=
sorry

end part1_part2_l508_508861


namespace sequence_product_permutation_l508_508039

-- Define the condition that n is a positive natural number (n ∈ ℕ∗)
def positive_nat (n : ℕ) : Prop := n > 0

-- State the mathematical problem
theorem sequence_product_permutation (n : ℕ) (h : positive_nat n) :
  (7 + n) * (8 + n) * (9 + n) * (10 + n) * (11 + n) * (12 + n) = nat.perm (12 + n) 6 :=
sorry

end sequence_product_permutation_l508_508039


namespace odd_function_iff_a2_b2_zero_l508_508587

noncomputable def f (x a b : ℝ) : ℝ := x * |x - a| + b

theorem odd_function_iff_a2_b2_zero {a b : ℝ} :
  (∀ x, f x a b = - f (-x) a b) ↔ a^2 + b^2 = 0 := by
  sorry

end odd_function_iff_a2_b2_zero_l508_508587


namespace ab_length_l508_508527

noncomputable def right_triangle_AB (BC AC : ℝ) (sqr_root : ℝ → ℝ) : ℝ := 
  let k := 24 / 13
  12 * k

theorem ab_length 
  (A B C : Type*)
  (AC : ℝ)
  (h_angleA : angle A B C = 90)
  (h_tanB : tan (angle B A C) = 5 / 12)
  (h_AC : AC = 24) :
  AB = 288 / 13 := 
sorry

end ab_length_l508_508527


namespace find_k1_k2_l508_508845

-- Definitions and conditions
def M : ℝ × ℝ := (-2, 0)
def ellipse (x y : ℝ) : Prop := x^2 + 2 * y^2 = 2
def line (P1 P2 : ℝ × ℝ) (k : ℝ) : Prop := ∃ (x1 y1 x2 y2 : ℝ), P1 = (x1, y1) ∧ P2 = (x2, y2) ∧ (y2 - y1) / (x2 - x1) = k
def midpoint (P1 P2 P : ℝ × ℝ) : Prop := ∃ (x1 y1 x2 y2 x y : ℝ), P1 = (x1, y1) ∧ P2 = (x2, y2) ∧ P = (x, y) ∧ x1 + x2 = 2 * x ∧ y1 + y2 = 2 * y

-- Theorem we need to prove
theorem find_k1_k2 (P1 P2 P : ℝ × ℝ) (k1 k2 : ℝ) (hk1 : k1 ≠ 0) 
  (hl : line P1 P2 k1) (he1 : ellipse (P1.1) (P1.2)) (he2 : ellipse (P2.1) (P2.2)) (hm : midpoint P1 P2 P) (osp : (P.2) / (P.1) = k2) :
  k1 * k2 = -1 / 2 :=
by
  sorry

end find_k1_k2_l508_508845


namespace length_minor_axis_l508_508378

-- Define the points given in the conditions
structure Point2D where
  x : ℝ
  y : ℝ

noncomputable def P1 : Point2D := ⟨0, 0⟩
noncomputable def P2 : Point2D := ⟨0, 4⟩
noncomputable def P3 : Point2D := ⟨6, 0⟩
noncomputable def P4 : Point2D := ⟨6, 4⟩
noncomputable def P5 : Point2D := ⟨-3, 2⟩

-- State that no three points among the given five are collinear
axiom no_three_collinear (a b c : Point2D) (h : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  ¬ Collinear ℝ (Set.univ : Set ({P // P ∈ {P1, P2, P3, P4, P5}}))

-- Unique conic section passing through the given points
axiom unique_conic_section (pts : Set Point2D) (h : pts = {P1, P2, P3, P4, P5}) :
  ∃! f : Point2D → ℝ, is_conic_section f ∧ ∀ P ∈ pts, f P = 0

-- The length of the minor axis of the ellipse with center (3,2), and axes parallel to coordinate axes
def is_ellipse (f : Point2D → ℝ) : Prop :=
  ∀ P : Point2D, f P = (P.x - 3)^2 / 36 + (P.y - 2)^2 / (16/3) - 1

-- The statement to be proved
theorem length_minor_axis :
  ∀ f : Point2D → ℝ, is_ellipse f → (∃ b : ℝ, b = 4 * sqrt 3 / 3 ∧ 2 * b = 8 * sqrt 3 / 3) :=
sorry

end length_minor_axis_l508_508378


namespace division_quotient_l508_508179

-- Define conditions
def dividend : ℕ := 686
def divisor : ℕ := 36
def remainder : ℕ := 2

-- Define the quotient
def quotient : ℕ := dividend - remainder

theorem division_quotient :
  quotient = divisor * 19 :=
sorry

end division_quotient_l508_508179


namespace bus_speed_excluding_stoppages_l508_508415

-- Define the conditions
def speed_including_stoppages : ℝ := 48 -- km/hr
def stoppage_time_per_hour : ℝ := 15 / 60 -- 15 minutes is 15/60 hours

-- The main theorem stating what we need to prove
theorem bus_speed_excluding_stoppages : ∃ v : ℝ, (v * (1 - stoppage_time_per_hour) = speed_including_stoppages) ∧ v = 64 :=
begin
  sorry,
end

end bus_speed_excluding_stoppages_l508_508415


namespace range_of_x8_l508_508095

-- Sequence definition
def sequence (x : ℕ → ℕ) : Prop := ∀ n ≥ 1, x (n + 2) = x (n + 1) + x n

-- Given conditions
variables {x : ℕ → ℕ} (h1 : 0 ≤ x 1) (h2 : x 1 ≤ x 2)
noncomputable def x1 := 1 -- for using later to evaluate
noncomputable def x2 := 2-- for using later to evaluate
noncomputable def x8_function := (13 * x 2) + (8 * x 1)

theorem range_of_x8 {x : ℕ → ℕ}
  (h1 : 0 ≤ x 1) 
  (h2 : x 1 ≤ x 2)
  (hSeq : sequence x )
  (hx7 : 1 ≤ x 7 ∧ x 7 ≤ 2) :
  21/13 ≤ x8 ∧ x8 ≤ 13/4:= 
  sorry

end range_of_x8_l508_508095


namespace find_k_value_l508_508907

theorem find_k_value
  (x y k : ℝ)
  (h1 : 4 * x + 3 * y = 1)
  (h2 : k * x + (k - 1) * y = 3)
  (h3 : x = y) :
  k = 11 :=
  sorry

end find_k_value_l508_508907


namespace greg_pages_per_day_l508_508876

variable (greg_pages : ℕ)
variable (brad_pages : ℕ)

theorem greg_pages_per_day :
  brad_pages = 26 → brad_pages = greg_pages + 8 → greg_pages = 18 :=
by
  intros h1 h2
  rw [h1, add_comm] at h2
  linarith

end greg_pages_per_day_l508_508876


namespace values_of_xyz_l508_508314

theorem values_of_xyz (x y z : ℝ) (h1 : 2 * x - y + z = 14) (h2 : y = 2) (h3 : x + z = 3 * y + 5) : 
  x = 5 ∧ y = 2 ∧ z = 6 := 
by
  sorry

end values_of_xyz_l508_508314


namespace part1_union_part1_intersect_complement_part2_range_a_l508_508832

open Set

-- Problem Setup
namespace ProofProblem

variables (x a : ℝ) (U : Set ℝ := univ)
def A := {x : ℝ | 2 ≤ x ∧ x ≤ 8}
def B := {x : ℝ | 1 < x ∧ x < 6}
def C (a : ℝ) := {x : ℝ | x > a}

-- Proof Statements

theorem part1_union : A ∪ B = {x : ℝ | 1 < x ∧ x ≤ 8} := 
by sorry

theorem part1_intersect_complement : (U \ A) ∩ B = {x : ℝ | 1 < x ∧ x < 2} :=
by sorry

theorem part2_range_a (h : (A ∩ C a) ≠ ∅) : a < 8 :=
by sorry

end ProofProblem

end part1_union_part1_intersect_complement_part2_range_a_l508_508832


namespace puppy_cost_is_10_l508_508238

def puppy_cost : ℕ :=
  let cups_per_day := 1 / 3
  let days := 3 * 7
  let total_cups := days * cups_per_day
  let bag_size := 3.5
  let bag_cost := 2
  let total_cost := 14
  let bags_needed := total_cups / bag_size
  total_cost - (bags_needed * bag_cost)

theorem puppy_cost_is_10 : puppy_cost = 10 := by
  sorry

end puppy_cost_is_10_l508_508238


namespace coefficient_of_x_squared_l508_508543

noncomputable theory

open_locale big_operators

theorem coefficient_of_x_squared :
  let term (r : ℕ) := (-(1/2))^r * (Nat.choose 5 r : ℚ) * x^((10 - 3 * r) / 2) in
  (∑ r in Finset.range 6, term r).coeff 2 = (5 / 2 : ℚ) := 
sorry

end coefficient_of_x_squared_l508_508543


namespace difference_of_squares_65_35_l508_508762

theorem difference_of_squares_65_35 : 65^2 - 35^2 = 3000 := 
  sorry

end difference_of_squares_65_35_l508_508762


namespace integer_count_between_sqrt8_and_sqrt78_l508_508151

theorem integer_count_between_sqrt8_and_sqrt78 :
  ∃ (n : ℕ), n = 6 ∧ ∀ (x : ℤ), (⌈Real.sqrt 8⌉ ≤ x ∧ x ≤ ⌊Real.sqrt 78⌋) ↔ (3 ≤ x ∧ x ≤ 8) := by
  sorry

end integer_count_between_sqrt8_and_sqrt78_l508_508151


namespace count_intersecting_chord_sets_l508_508598

-- Define the problem using Lean theorem

theorem count_intersecting_chord_sets 
  (points : Finset ℕ) -- Finite set of points on the circle
  (h_points : points.card = 20) : -- There are exactly 20 points
  ∃ n, n = 156180 := -- The number of sets of intersecting chords is 156180
begin
  sorry
end

end count_intersecting_chord_sets_l508_508598


namespace twenty_fourth_digit_of_sum_l508_508657

theorem twenty_fourth_digit_of_sum : 
  let dec_1_9 := "0." ++ String.repeat "1" 1000 in
  let dec_1_4 := "0.250000" in
  let sum := "0.361111" in
  (String.get sum 5) = '1' := 
sorry

end twenty_fourth_digit_of_sum_l508_508657


namespace bus_speed_excluding_stoppages_l508_508413

theorem bus_speed_excluding_stoppages (v : ℝ) (stoppage_time : ℝ) (speed_incl_stoppages : ℝ) :
  stoppage_time = 15 / 60 ∧ speed_incl_stoppages = 48 → v = 64 :=
by
  intro h
  sorry

end bus_speed_excluding_stoppages_l508_508413


namespace line_passes_through_point_outside_plane_l508_508676

variable (P : Type) (a β : Set P) [SetTheory P]

def passes_through_point_outside_plane (P : P) (a : Set P) (β : Set P) [SetTheory P] : Prop :=
  P ∈ a ∧ P ∉ β

theorem line_passes_through_point_outside_plane (P : P) (a : Set P) (β : Set P) [SetTheory P] :
  passes_through_point_outside_plane P a β ↔ (P ∈ a ∧ P ∉ β) :=
by 
  sorry

end line_passes_through_point_outside_plane_l508_508676


namespace clever_cats_event_l508_508090

def is_multiple (n m : ℕ) : Prop := ∃ k : ℕ, n = m * k

def count_multiple_upto (m n : ℕ) : ℕ := n / m

theorem clever_cats_event :
  let total_fans := 5000
  let beverage_coupon := 90
  let snack_coupon := 45
  let hat_coupon := 60
  let tshirt_coupon := 100
  let lcm_900 := Nat.lcm beverage_coupon (Nat.lcm snack_coupon (Nat.lcm hat_coupon tshirt_coupon))
  count_multiple_upto lcm_900 total_fans = 5 :=
by
  let total_fans := 5000
  let beverage_coupon := 90
  let snack_coupon := 45
  let hat_coupon := 60
  let tshirt_coupon := 100
  let lcm_900 := Nat.lcm beverage_coupon (Nat.lcm snack_coupon (Nat.lcm hat_coupon tshirt_coupon))
  have h_lcm: lcm_900 = 900 := sorry
  have h_count: count_multiple_upto lcm_900 total_fans = 5000 / 900 := sorry
  rw h_count
  norm_num
  rw h_lcm
  norm_num


end clever_cats_event_l508_508090


namespace equation_of_circle_tangent_passing_conditions_l508_508421

noncomputable def is_tangent_to_line (x y r : ℝ) := r = (|x + y - 1|) / (real.sqrt (1^2 + 1^2))
noncomputable def is_on_circle (x y a b r : ℝ) := (x - a)^2 + (y - b)^2 = r^2
noncomputable def on_line_y_eq_neg2x (a b : ℝ) := b = -2 * a

theorem equation_of_circle_tangent_passing_conditions :
  ∃ (a b r : ℝ), (is_tangent_to_line a b r) ∧ (is_on_circle 2 (-1) a b r) ∧ (on_line_y_eq_neg2x a b) ∧ 
                  (∀ x y, (x - a)^2 + (y - b)^2 = r^2 ↔ (x - 1)^2 + (y + 2)^2 = 2) := 
by
  sorry

end equation_of_circle_tangent_passing_conditions_l508_508421


namespace odd_even_deriv_signs_l508_508110

variables {R : Type} [OrderedRing R] 
  {f g : R → R}

theorem odd_even_deriv_signs (f_odd : ∀ x, f (-x) = -f (x))
  (g_even : ∀ x, g (-x) = g (x))
  (f_prime_pos : ∀ {x : R}, 0 < x → f' x > 0)
  (g_prime_pos : ∀ {x : R}, 0 < x → g' x > 0) :
  ∀ {x : R}, x < 0 → f' x > 0 ∧ g' x < 0 :=
by
  sorry

end odd_even_deriv_signs_l508_508110


namespace commission_percentage_l508_508965

theorem commission_percentage (fixed_salary second_base_salary sales_amount earning: ℝ) (commission: ℝ) 
  (h1 : fixed_salary = 1800)
  (h2 : second_base_salary = 1600)
  (h3 : sales_amount = 5000)
  (h4 : earning = 1800) :
  fixed_salary = second_base_salary + (sales_amount * commission) → 
  commission * 100 = 4 :=
by
  -- proof goes here
  sorry

end commission_percentage_l508_508965


namespace total_amount_spent_l508_508402

def price_of_brand_X_pen : ℝ := 4.00
def price_of_brand_Y_pen : ℝ := 2.20
def total_pens_purchased : ℝ := 12
def brand_X_pens_purchased : ℝ := 6

theorem total_amount_spent :
  let brand_X_cost := brand_X_pens_purchased * price_of_brand_X_pen
  let brand_Y_pens_purchased := total_pens_purchased - brand_X_pens_purchased
  let brand_Y_cost := brand_Y_pens_purchased * price_of_brand_Y_pen
  brand_X_cost + brand_Y_cost = 37.20 :=
by
  sorry

end total_amount_spent_l508_508402


namespace scientific_notation_11580000_l508_508912

theorem scientific_notation_11580000 :
  (11580000 : ℝ) = 1.158 * 10^7 :=
sorry

end scientific_notation_11580000_l508_508912


namespace pyramid_volume_l508_508997

noncomputable def volume_of_pyramid (S KO : ℝ) : ℝ :=
  (S * KO) / 3

theorem pyramid_volume (K O : Point) (S KO : ℝ) (α : ℝ) :
  S = base_area_projection β (base K O) KO → 
  volume_of_pyramid S KO = (S * KO) / 3 :=
by
  sorry

end pyramid_volume_l508_508997


namespace b_negative_of_conditions_l508_508727

variable {a b : ℝ}

-- Conditions
def two_pos_two_neg (x y z w : ℝ) : Prop :=
  ((x > 0) + (y > 0) + (z > 0) + (w > 0) = 2) ∧
  ((x < 0) + (y < 0) + (z < 0) + (w < 0) = 2)

-- Theorem
theorem b_negative_of_conditions (h : two_pos_two_neg (a + b) (a - b) (a * b) (a / b)) : b < 0 := 
  sorry

end b_negative_of_conditions_l508_508727


namespace area_of_triangle_PEQ_l508_508837

-- Define the conditions
def parabola_Focus : Type := { F : ℝ × ℝ // F = (1/2, 0) }
def parabola_C (x y : ℝ) : Prop := y^2 = 2 * x

def ray_l (x y : ℝ) : Prop := x = -1/2 ∧ y ≥ 0

def perpendicular_bisector_EF (E F Q P : ℝ × ℝ) : Prop :=
  (Q.1 = -1/2 ∧ Q.2 = 3/4) ∧ E.1 = -1/2 ∧ F = (1/2, 0) ∧
  (E.2 ≥ 0) ∧ 
  let M := ((E.1 + F.1) / 2, (E.2 + F.2) / 2) in
  (P.2 = (1/2) * P.1 + 1) ∧
  parabola_C P.1 P.2

-- Define the proof problem statement
theorem area_of_triangle_PEQ :
  ∀ (E Q P : ℝ × ℝ) (F : parabola_Focus),
  (ray_l E.1 E.2) ∧ (perpendicular_bisector_EF E F.1 Q P) →
  let area := 1/2 * dist E Q * dist E P in
  area = 5/4 :=
by
  sorry

end area_of_triangle_PEQ_l508_508837


namespace calc_pow_expression_l508_508040

theorem calc_pow_expression : (27^3 * 9^2) / 3^15 = 1 / 9 := 
by sorry

end calc_pow_expression_l508_508040


namespace area_triangle_DEF_l508_508067

-- Define the basic setup of the problem using provided conditions
def is_center_of_adjacent_squares (D E F : ℝ × ℝ) :=
  (D = (0, 2) ∧ E = (2, 2)) ∨ (E = (0, 2) ∧ F = (2, 2))

-- Main theorem statement: the calculation of the area of triangle DEF
theorem area_triangle_DEF (D E F : ℝ × ℝ) (h_adjacent : is_center_of_adjacent_squares D E ∧ is_center_of_adjacent_squares E F) :
  let side1 := 2
  let hypotenuse := 2 * real.sqrt 2
  let area := 1 / 2 * side1 * side1
  area = 2 :=
sorry

end area_triangle_DEF_l508_508067


namespace quadrilateral_cosine_law_proof_l508_508934

variables (ABCD : Type)
variables (A B C D P : ABCD)
variables (a b c d e f : ℝ)
variables (angle_APB P_angle : ℝ)
variables (AB : A ≠ B) (BC : B ≠ C) (CD : C ≠ D) (DA : D ≠ A)
variables (AC_eq_e : dist A C = e) (BD_eq_f : dist B D = f)
variables (AB_eq_a : dist A B = a) (BC_eq_b : dist B C = b) (CD_eq_c : dist C D = c) (DA_eq_d : dist D A = d)
variables (angle_APB_eq_P : angle A P B = P_angle)

theorem quadrilateral_cosine_law_proof :
  2 * e * f * Real.cos P_angle = - (a^2) + (b^2) - (c^2) + (d^2) := 
sorry

end quadrilateral_cosine_law_proof_l508_508934


namespace solution_set_of_inequality_l508_508287

theorem solution_set_of_inequality :
  {x : ℝ | x^2 - 5 * x + 6 ≤ 0} = set.Icc 2 3 :=
begin
  sorry
end

end solution_set_of_inequality_l508_508287


namespace sin_2alpha_value_l508_508906

noncomputable def sin_2alpha_through_point (x y : ℝ) : ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let sin_alpha := y / r
  let cos_alpha := x / r
  2 * sin_alpha * cos_alpha

theorem sin_2alpha_value :
  sin_2alpha_through_point (-3) 4 = -24 / 25 :=
by
  sorry

end sin_2alpha_value_l508_508906


namespace eccentricity_of_hyperbola_l508_508486

noncomputable def a : ℝ := 4
noncomputable def b : ℝ := 3
noncomputable def c : ℝ := 5

def hyperbola_eq (x y : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = 1

def focus := (5, 0)

def eccentricity (c a : ℝ) : ℝ := c / a

theorem eccentricity_of_hyperbola : 
  ∀ (a b c : ℝ), a > 0 → b = 3 → c = 5 → eccentricity c a = 5 / 4 :=
by
  intros
  sorry

end eccentricity_of_hyperbola_l508_508486


namespace economic_model_l508_508363

theorem economic_model :
  ∃ (Q_s : ℝ → ℝ) (T t_max T_max : ℝ),
  (∀ P : ℝ, Q_d P = 688 - 4 * P) ∧
  (∀ P_e Q_e : ℝ, 1.5 * (4 * P_e / Q_e) = (Q_s'.eval P_e / Q_e)) ∧
  (Q_s 64 = 72) ∧
  (∀ P : ℝ, Q_s P = 6 * P - 312) ∧
  (T = 6480) ∧
  (t_max = 60) ∧
  (T_max = 8640)
where 
  Q_d: ℝ → ℝ := λ P, 688 - 4 * P
  Q_s'.eval : ℝ → ℝ := sorry

end economic_model_l508_508363


namespace sum_a10_a11_l508_508942

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∃ q : ℝ, (∀ n : ℕ, a (n + 1) = a n * q)

theorem sum_a10_a11 {a : ℕ → ℝ} (h_seq : geometric_sequence a)
  (h1 : a 1 + a 2 = 2)
  (h4 : a 4 + a 5 = 4) :
  a 10 + a 11 = 16 :=
by {
  sorry
}

end sum_a10_a11_l508_508942


namespace find_base_k_representation_l508_508443

theorem find_base_k_representation :
  ∃ k : ℕ, (k > 0) ∧ (0.363636..._k = ((∑ (n : ℕ), (3*(k^n*(2 - n%2)) / (k^(n+1)))) where k = 30))  := sorry

end find_base_k_representation_l508_508443


namespace tank_capacity_l508_508157

theorem tank_capacity :
  ∀ (T : ℚ), (3 / 4) * T + 4 = (7 / 8) * T → T = 32 :=
by
  intros T h
  sorry

end tank_capacity_l508_508157


namespace scientific_notation_11580000_l508_508915

theorem scientific_notation_11580000 :
  11580000 = 1.158 * 10^7 :=
sorry

end scientific_notation_11580000_l508_508915


namespace right_triangle_3_4_5_l508_508723

theorem right_triangle_3_4_5 :
  ∀ (a b c : ℕ), (a = 3 ∧ b = 4 ∧ c = 5) → a^2 + b^2 = c^2 :=
by
  intros a b c h
  cases h with ha hb
  cases hb with hb hc
  rw [ha, hb, hc]
  sorry

end right_triangle_3_4_5_l508_508723


namespace solve_inequality_l508_508612

theorem solve_inequality :
  {x : ℝ | 0 ≤ x^2 - x - 2 ∧ x^2 - x - 2 ≤ 4} = {x : ℝ | (-2 ≤ x ∧ x ≤ -1) ∨ (2 ≤ x ∧ x ≤ 3)} :=
sorry

end solve_inequality_l508_508612


namespace problem_1_problem_2_l508_508440

theorem problem_1 (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h1 : ∀ n : ℕ, 0 < n → 2 * S n = (a n)^2 + a n):
  (∀ n : ℕ, 0 < n → a n = n) := 
sorry

theorem problem_2 (a : ℕ → ℕ) (b : ℕ → ℕ) (T : ℕ → ℕ)
  (h1 : ∀ n : ℕ, 0 < n → 2 * ((λ s, ∑ i in finset.range s, a (i + 1)) n) = (a n)^2 + a n)
  (h2 : ∀ n : ℕ, b n = ∑ i in finset.range n, a (i + 1) * 3^(i - n))
  (h3 : ∀ n : ℕ, a n = n) :
  (∀ n : ℕ, T n = ∑ i in finset.range n, b (i + 1) → T n = n ^ 2 / 4 + 1 / 8 * (1 - 3 ^ (-n))) := 
sorry

end problem_1_problem_2_l508_508440


namespace range_of_f_values_of_a_b_l508_508122

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 3 / 2) * Real.sin (2 * x) - Real.cos x ^ 2 - 1 / 2

theorem range_of_f :
  set.range (λ x, f x) = set.Icc (-1 - Real.sqrt 3 / 2) 0 :=
sorry

variables {A B C a b c : ℝ}

theorem values_of_a_b (h1 : c = Real.sqrt 3)
  (h2 : sin (2 * C - π / 6) = 1)
  (h3 : 0 < C ∧ C < π)
  (h4 : 1 + Real.sin A = λ x:ℝ, 2 + Real.sin B) :
  a = 1 ∧ b = 2 :=
sorry

end range_of_f_values_of_a_b_l508_508122


namespace poly_divides_diff_l508_508558

noncomputable def poly_cond (f : ℝ → ℝ) (n : ℕ) : Prop :=
  ∀ (x y : ℝ), 0 ≤ x → x < y → y ≤ n → (f(x) - f(y)) / (x - y) ∈ ℤ

theorem poly_divides_diff {f : ℝ → ℝ} {n : ℕ} 
  (h₁ : polynomial.degree f = n) 
  (h₂ : poly_cond f n) 
  (a b : ℤ) 
  (h₃ : a ≠ b) : 
  (a - b) ∣ (f a - f b) := 
sorry

end poly_divides_diff_l508_508558


namespace trajectory_of_C_is_ellipse_l508_508289

theorem trajectory_of_C_is_ellipse :
  ∀ (C : ℝ × ℝ),
  ((C.1 + 4)^2 + C.2^2).sqrt + ((C.1 - 4)^2 + C.2^2).sqrt = 10 →
  (C.2 ≠ 0) →
  (C.1^2 / 25 + C.2^2 / 9 = 1) :=
by
  intros C h1 h2
  sorry

end trajectory_of_C_is_ellipse_l508_508289


namespace impossible_non_self_intersecting_path_rubiks_cube_l508_508949

theorem impossible_non_self_intersecting_path_rubiks_cube :
  ¬∃ f : Fin 54 → (Fin 56 × Fin 56), 
    (∀ i : Fin 54, f i.1 ≠ f i.2) ∧ 
    (∀ i j : Fin 54, i ≠ j → f i ≠ f j ∧ f i.1 ≠ f j.1 ∧ f i.2 ≠ f j.2) :=
by sorry

end impossible_non_self_intersecting_path_rubiks_cube_l508_508949


namespace range_a_increasing_function_l508_508126

def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then 2^x + a else a * x^2 + 2 * x

theorem range_a_increasing_function :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) ↔ (a ∈ set.Icc (-1 : ℝ) (0 : ℝ)) := by
  sorry

end range_a_increasing_function_l508_508126


namespace noah_ava_zoo_trip_l508_508243

theorem noah_ava_zoo_trip :
  let tickets_cost := 5
  let bus_fare := 1.5
  let initial_money := 40
  let num_people := 2
  let round_trip := 2

  initial_money - (num_people * tickets_cost + num_people * round_trip * bus_fare) = 24 :=
by
  let tickets_cost := 5
  let bus_fare := 1.5
  let initial_money := 40
  let num_people := 2
  let round_trip := 2

  have cost := num_people * tickets_cost + num_people * round_trip * bus_fare
  have remaining := initial_money - cost
  have : remaining = 24, sorry
  exact this

end noah_ava_zoo_trip_l508_508243


namespace estimated_red_light_runners_l508_508652

noncomputable def survey : ℤ :=
  let total_students := 600
  let yes_responses := 180
  let half_students := total_students / 2
  let estimate_ran_red_light := (yes_responses - half_students / 2) * 2
  estimate_ran_red_light

theorem estimated_red_light_runners : survey = 60 := 
by 
  let total_students := 600
  let yes_responses := 180
  let half_students := total_students / 2
  let estimate_ran_red_light := (yes_responses - half_students / 2) * 2
  show estimate_ran_red_light = 60 from sorry

end estimated_red_light_runners_l508_508652


namespace restore_supply_function_determine_tax_revenue_determine_optimal_tax_rate_determine_maximum_tax_revenue_l508_508364

-- Defining the given conditions
noncomputable def market_demand (P : ℝ) : ℝ := 688 - 4 * P
noncomputable def post_tax_producer_price : ℝ := 64
noncomputable def per_unit_tax : ℝ := 90
noncomputable def elasticity_supply_no_tax (P_e : ℝ) (Q_e : ℝ) : ℝ :=
  1.5 * (-(4 * P_e / Q_e))

-- Supply function to be proven
noncomputable def supply_function (P : ℝ) : ℝ := 6 * P - 312

-- Total tax revenue to be proven
noncomputable def total_tax_revenue : ℝ := 6480

-- Optimal tax rate to be proven
noncomputable def optimal_tax_rate : ℝ := 60

-- Maximum tax revenue to be proven
noncomputable def maximum_tax_revenue : ℝ := 8640

-- Theorem statements that need to be proven
theorem restore_supply_function (P : ℝ) : 
  supply_function P = 6 * P - 312 := sorry

theorem determine_tax_revenue : 
  total_tax_revenue = 6480 := sorry

theorem determine_optimal_tax_rate : 
  optimal_tax_rate = 60 := sorry

theorem determine_maximum_tax_revenue : 
  maximum_tax_revenue = 8640 := sorry

end restore_supply_function_determine_tax_revenue_determine_optimal_tax_rate_determine_maximum_tax_revenue_l508_508364


namespace correct_product_of_a_and_b_l508_508538

theorem correct_product_of_a_and_b (a b : ℕ) (ha : a ≥ 100 ∧ a < 1000) (hb : b ≥ 10 ∧ b < 100) (hb_reverse : ∃ b', b' = 11 ∧ b' = num_reverse b) (h_eq : a * 11 = 143) :
  a * b = 143 :=
sorry

end correct_product_of_a_and_b_l508_508538


namespace lcm_of_18_30_45_l508_508694

theorem lcm_of_18_30_45 : Nat.lcm (Nat.lcm 18 30) 45 = 90 :=
by
  rw [Nat.lcm_assoc, Nat.lcm_comm (Nat.lcm 18 30), Nat.lcm_assoc, Nat.lcm_self, Nat.lcm_comm 30 45]
  { lcm_factors }
  sorry

end lcm_of_18_30_45_l508_508694


namespace prime_p_geq_5_div_24_l508_508895

theorem prime_p_geq_5_div_24 (p : ℕ) (hp : Nat.Prime p) (hp_geq_5 : p ≥ 5) : 24 ∣ (p^2 - 1) :=
sorry

end prime_p_geq_5_div_24_l508_508895


namespace calc_3a2008_minus_5b2008_l508_508874

theorem calc_3a2008_minus_5b2008 (a b : ℝ) (h1 : a - b = 1) (h2 : a^2 - b^2 = -1) : 3 * a ^ 2008 - 5 * b ^ 2008 = -5 :=
by
  sorry

end calc_3a2008_minus_5b2008_l508_508874


namespace _l508_508208

def count_trailing_zeros (n : ℕ) : ℕ :=
  if n = 0 then 0 else
    let rec count_factors (x d : ℕ) : ℕ :=
      if x % d ≠ 0 then 0 else 1 + count_factors (x / d) d;
    count_factors n 10

def product_of_factorials (n : ℕ) : ℕ :=
  (finset.range (n + 1)).prod factorial

def trailing_zeros_of_product : ℕ :=
  count_trailing_zeros (product_of_factorials 50)

example : trailing_zeros_of_product % 100 = 31 := 
by {
  -- This is a statement of the theorem (no proof included)
  sorry
}

end _l508_508208


namespace Patricia_earns_more_l508_508961

-- Define the function for compound interest with annual compounding
noncomputable def yearly_compound (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r)^n

-- Define the function for compound interest with quarterly compounding
noncomputable def quarterly_compound (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r / 4)^ (4 * n)

-- Define the conditions
def P := 50000.0
def r := 0.04
def n := 2

-- Define the amounts for Jose and Patricia using their respective compounding methods
def A_Jose := yearly_compound P r n
def A_Patricia := quarterly_compound P r n

-- Define the target difference in earnings
def difference := A_Patricia - A_Jose

-- Theorem statement
theorem Patricia_earns_more : difference = 63 := by
  sorry

end Patricia_earns_more_l508_508961


namespace hyperbola_eccentricity_l508_508131

-- Definitions of the conditions involved
variables (a b r : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) (h_r_pos : r > 0)
def hyperbola (x y : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = 1
def circle (x y : ℝ) : Prop := x^2 + y^2 = r^2
def foci_distance (e : ℝ) : ℝ := 2 * b * sqrt(1 + a^2 / b^2)

-- Main statement to prove
theorem hyperbola_eccentricity : 
  (∀ (x y : ℝ), hyperbola a b x y → circle r x y) → 
  (max (λ θ : ℝ, |sqrt(a^2 + b^2) * cos θ * (r + 1)|) / r) = 4 * sqrt(2) → 
  2 * sqrt(1 + (a^2 / b^2)) = 2 * sqrt(2) := 
sorry

end hyperbola_eccentricity_l508_508131


namespace problem_1_problem_2_problem_3_problem_4_l508_508052

theorem problem_1 (a b : ℝ) (h : a ≠ b) (h1 : a - b = 1) : a^3 - b^3 - ab - a^2 - b^2 = 0 :=
sorry

theorem problem_2 (a : ℝ) (f : ℝ → ℝ) (h : ∀ x, f(x) = (x - a) * (x + 2)) (heven : ∀ x, f(x) = f(-x)) : a ≠ -2 :=
sorry

theorem problem_3 (k : ℝ) : (∃ P Q : ℝ × ℝ, (x : P) ^ 2 + (y : P) ^ 2 - 2 * x = 0 ∧ (x : Q) ^ 2 + (y : Q) ^ 2 - 2 * x = 0 ∧ (2 * x - y + 2 = 0)) → k ≠ 2 :=
sorry

theorem problem_4 (θ : ℝ) (h : tan θ = 2) : cos (2 * θ) = -3 / 5 :=
sorry

end problem_1_problem_2_problem_3_problem_4_l508_508052


namespace change_given_l508_508745

-- Define the given conditions
def oranges_cost := 40
def apples_cost := 50
def mangoes_cost := 60
def initial_amount := 300

-- Calculate total cost of fruits
def total_fruits_cost := oranges_cost + apples_cost + mangoes_cost

-- Define the given change
def given_change := initial_amount - total_fruits_cost

-- Prove that the given change is equal to 150
theorem change_given (h_oranges : oranges_cost = 40)
                     (h_apples : apples_cost = 50)
                     (h_mangoes : mangoes_cost = 60)
                     (h_initial : initial_amount = 300) :
  given_change = 150 :=
by
  -- Proof is omitted, indicated by sorry
  sorry

end change_given_l508_508745


namespace trigonometric_identity_l508_508106

theorem trigonometric_identity (α x : ℝ) (h₁ : 5 * Real.cos α = x) (h₂ : x ^ 2 + 16 = 25) (h₃ : α > Real.pi / 2 ∧ α < Real.pi):
  x = -3 ∧ Real.tan α = -4 / 3 :=
by
  sorry

end trigonometric_identity_l508_508106


namespace B_gives_C_100_meters_start_l508_508931

-- Definitions based on given conditions
variables (Va Vb Vc : ℝ) (T : ℝ)

-- Assume the conditions based on the problem statement
def race_condition_1 := Va = 1000 / T
def race_condition_2 := Vb = 900 / T
def race_condition_3 := Vc = 850 / T

-- Theorem stating that B can give C a 100 meter start
theorem B_gives_C_100_meters_start
  (h1 : race_condition_1 Va T)
  (h2 : race_condition_2 Vb T)
  (h3 : race_condition_3 Vc T) :
  (Vb = (1000 - 100) / T) :=
by
  -- Utilize conditions h1, h2, and h3
  sorry

end B_gives_C_100_meters_start_l508_508931


namespace day_of_300th_day_of_previous_year_is_monday_l508_508518

-- Define a function that converts a day of the week number (1=Monday, 2=Tuesday, ..., 7=Sunday)
-- to the previous day of the week.
def prev_day (day : ℕ) : ℕ :=
  if day = 1 then 7 else day - 1

-- Define a function that calculates the day of the week after moving n days back from a given day.
def days_back (start_day : ℕ) (n : ℕ) : ℕ :=
  nat.rec_on n start_day (λ _ prev, prev_day prev)

theorem day_of_300th_day_of_previous_year_is_monday
  (currentYear : ℕ)
  (nextYear : ℕ)
  (previousYear : ℕ)
  (day_of_week : ℕ → ℕ)
  (sunday : ℕ)
  (h1 : day_of_week (currentYear * 365 + 200) = sunday)
  (h2 : day_of_week ((currentYear + 1) * 365 + 100) = sunday) :
  day_of_week ((currentYear - 1) * 365 + 300) = 1 :=
by
  sorry

end day_of_300th_day_of_previous_year_is_monday_l508_508518


namespace first_folder_number_l508_508033

theorem first_folder_number (stickers : ℕ) (folders : ℕ) : stickers = 999 ∧ folders = 369 → 100 = 100 :=
by sorry

end first_folder_number_l508_508033


namespace find_f_of_3_l508_508570

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_of_3.5 (f : ℝ → ℝ) 
  (h1 : ∀ x, f (-x) = f x) 
  (h2 : ∀ x, f (x + 2) + f x = 0) 
  (h3 : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = x) :
  f 3.5 = 0.5 :=
sorry

end find_f_of_3_l508_508570


namespace binary_sum_to_decimal_l508_508397

theorem binary_sum_to_decimal :
  let bin1 := "1101011"
  let bin2 := "1010110"
  let dec1 := 64 + 32 + 0 + 8 + 0 + 2 + 1 -- decimal value of "1101011"
  let dec2 := 64 + 0 + 16 + 0 + 4 + 2 + 0 -- decimal value of "1010110"
  dec1 + dec2 = 193 := by
  sorry

end binary_sum_to_decimal_l508_508397


namespace number_of_valid_numbers_l508_508636

-- Define a function that checks if a number is composed of digits from the set {1, 2, 3}
def composed_of_123 (n : ℕ) : Prop :=
  ∀ (d : ℕ), d ∈ n.digits 10 → d = 1 ∨ d = 2 ∨ d = 3

-- Define a predicate for a number being less than 200,000
def less_than_200000 (n : ℕ) : Prop := n < 200000

-- Define a predicate for a number being divisible by 3
def divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

-- The main theorem statement
theorem number_of_valid_numbers : ∃ (count : ℕ), count = 202 ∧ 
  (∀ (n : ℕ), less_than_200000 n → composed_of_123 n → divisible_by_3 n → n < count) :=
sorry

end number_of_valid_numbers_l508_508636


namespace number_of_divisors_of_square_is_odd_l508_508889

theorem number_of_divisors_of_square_is_odd (x : ℕ) (h : ∃ n : ℕ, x = n^2) :
  ∃ d : ℕ, d = (∏ i in (range (nat.prime_factorization x).card), 2 * (nat.prime_factorization x) i + 1) ∧ (d = 51 ∨ d = 53) :=
by {
  sorry
}

end number_of_divisors_of_square_is_odd_l508_508889


namespace calculation_correct_l508_508391

theorem calculation_correct : (18 / (3 + 9 - 6)) * 4 = 12 :=
by
  sorry

end calculation_correct_l508_508391


namespace percentage_reduction_l508_508700

variable (C S newS newC : ℝ)
variable (P : ℝ)
variable (hC : C = 50)
variable (hS : S = 1.25 * C)
variable (hNewS : newS = S - 10.50)
variable (hGain30 : newS = 1.30 * newC)
variable (hNewC : newC = C - P * C)

theorem percentage_reduction (C S newS newC : ℝ) (hC : C = 50) 
  (hS : S = 1.25 * C) (hNewS : newS = S - 10.50) 
  (hGain30 : newS = 1.30 * newC) 
  (hNewC : newC = C - P * C) : 
  P = 0.20 :=
by
  sorry

end percentage_reduction_l508_508700


namespace min_value_and_range_of_a_l508_508985

/-- Given the function f(x) = 2 * |x - 2| - x + 5.
1. Prove that the minimum value of the function f(x) is 3.
2. Prove that if the inequality |x - a| + |x + 2| ≥ 3 always holds, then the range of the real number a is a ≤ -5 or a ≥ 1.
-/
theorem min_value_and_range_of_a (f : ℝ → ℝ) (m : ℝ) (a : ℝ) :
  (∀ x : ℝ, f x = 2 * |x - 2| - x + 5) →
  (∀ x : ℝ, 2 * |x - 2| - x + 5 ≥ m) → 
  (m = 3) →
  (∀ x : ℝ, 0 ≤ |(x - a) - (x + 2)|) →
  (∀ x : ℝ, |x - a| + |x + 2| ≥ m) → 
  (∀ x : ℝ, |a + 2| ≥ m) → 
  (a ≤ -5 ∨ a ≥ 1) :=
by
  assume h1 h2 h3 h4 h5 h6,
  sorry

end min_value_and_range_of_a_l508_508985


namespace not_a_CDF_l508_508119

def F (x : ℝ) : ℝ :=
  if x ≤ 0 then 0 else if x ≤ 2 then x^2 else 1

theorem not_a_CDF : ¬(∀ x y : ℝ, x ≤ y → F x ≤ F y) ∨ 
                    F (-1) ≠ 0 ∨ F 3 ≠ 1 ∨ 
                    ¬(∀ x : ℝ, ∃ δ > 0, ∀ ε, 0 < ε < δ → F(x + ε) = F x) ∨ 
                    ¬(∀ x: ℝ, 0 ≤ F x ∧ F x ≤ 1) :=
by
  sorry

end not_a_CDF_l508_508119


namespace equation_of_parabola_area_function_equal_areas_exist_l508_508368

-- Given conditions
def parabola_C (p : ℝ) (p_pos : p > 0) : set (ℝ × ℝ) := {P | ∃ x y, P = (x, y) ∧ x^2 = 2 * p * y}
def point_Q (m : ℝ) : ℝ × ℝ := (m, 1 / 2)
def distance_to_focus (Q : ℝ × ℝ) (p : ℝ) := ((Q.2 - (-p / 2)).abs + p / 2 = 1)
def line_through_M (k : ℝ) : set (ℝ × ℝ) := {P | ∃ x y, P = (x, y) ∧ y = k * x + 2}
def points_AB (parabola_pt : ℝ → ℝ × ℝ) : set (ℝ × ℝ) :=
  {P | ∃ n > 0, P = (n, n^2 / 2)}

-- Proof goals
theorem equation_of_parabola :
  ∀ (p : ℝ) (p_pos : p > 0) (Q : ℝ × ℝ) (m : ℝ) (Q_m : Q = point_Q m),
    distance_to_focus Q p → p = 1 :=
sorry

theorem area_function :
  ∀ (n : ℕ), n > 0 → 
    (∀ k : ℝ, l = line_through_M k → 
      let A := (n, n^2 / 2) in let B := (2 * k - n, (2 * k - n)^2 / 2) in
        f(n) = n + 4 / n) :=
sorry

theorem equal_areas_exist :
  ∀ (m n : ℕ), m > 0 ∧ n > 0 ∧ m ≠ n →
    (f(m) = f(n)) → 
      (m = 1 ∧ n = 4) ∨ (m = 4 ∧ n = 1) :=
sorry

end equation_of_parabola_area_function_equal_areas_exist_l508_508368


namespace number_of_goose_eggs_laid_l508_508336

noncomputable def total_goose_eggs := 
  let hatched_fraction := (1 : ℚ) / 2
  let survived_first_month_fraction := (3 : ℚ) / 4
  let not_survived_first_year_fraction := (3 : ℚ) / 5
  let survived_first_year_fraction := (2 : ℚ) / 5
  let survived_first_year_geese := 120
  let expression := survived_first_year_fraction * survived_first_month_fraction * hatched_fraction
  400

theorem number_of_goose_eggs_laid (hatched_fraction survived_first_month_fraction survived_first_year_fraction survived_first_year_geese : ℚ) (total_goose_eggs : ℚ) : 
  hatched_fraction = 1 / 2 →
  survived_first_month_fraction = 3 / 4 →
  survived_first_year_fraction = 2 / 5 →
  survived_first_year_geese = 120 →
  total_goose_eggs = survived_first_year_geese / (survived_first_year_fraction * survived_first_month_fraction * hatched_fraction) →
  total_goose_eggs = 400 :=
by
  intro h1 h2 h3 h4 h5
  have h6 : total_goose_eggs = 400 := rfl
  exact h6

#eval number_of_goose_eggs_laid 1/2 3/4 2/5 120 400

end number_of_goose_eggs_laid_l508_508336


namespace sugar_percentage_correct_l508_508995

def percentage_sugar_in_mixture (initial_sugar_oz : ℝ) (spilled_sugar_oz : ℝ) (flour_g : ℝ) (conversion_factor : ℝ) : ℝ :=
  let remaining_sugar_oz := initial_sugar_oz - spilled_sugar_oz
  let remaining_sugar_g := remaining_sugar_oz * conversion_factor
  let total_mixture_g := remaining_sugar_g + flour_g
  (remaining_sugar_g / total_mixture_g) * 100

theorem sugar_percentage_correct :
  percentage_sugar_in_mixture 9.8 5.2 150 28.35 = 46.51 :=
by
  sorry

end sugar_percentage_correct_l508_508995


namespace ship_travel_time_l508_508019

noncomputable def travel_time (distance speed : ℝ) : ℝ :=
  distance / speed

theorem ship_travel_time :
  let b := 37 + 30 / 60 
  let l := 12 + 40 / 60
  let b1 := 31 + 30 / 60
  let l1 := 33 + 8 / 60
  let distance := 1987.5
  let speed := 18.5 in
  travel_time distance speed = 107.4324 := 
by
  sorry

end ship_travel_time_l508_508019


namespace statement_c_is_false_l508_508768

def heartsuit (x y : ℝ) : ℝ :=
  |x - y|

theorem statement_c_is_false : ¬ ∀ x y z : ℝ, heartsuit (heartsuit x y) (heartsuit y z) = |x - z| :=
by {
  -- We need to provide a counterexample to statement (C)
  let x := 0,
  let y := 1,
  let z := 2,
  have h1 : heartsuit x y = |x - y| := rfl,
  have h2 : heartsuit y z = |y - z| := rfl,
  have h3 : heartsuit (heartsuit x y) (heartsuit y z) = |heartsuit x y - heartsuit y z| := rfl,
  have h4 : heartsuit (heartsuit x y) (heartsuit y z) = |heartsuit x y - heartsuit y z| := by rw h3,
  have h5 : heartsuit (heartsuit 0 1) (heartsuit 1 2) = |1 - 1| := rfl,
  have h6 : heartsuit 0 2 = 2 := rfl,
  have h7 : heartsuit (heartsuit (0 : ℝ) (1 : ℝ)) (heartsuit (1 : ℝ) (2 : ℝ)) ≠ heartsuit (0 : ℝ) (2 : ℝ) by sorry,
  exact h7
}

end statement_c_is_false_l508_508768


namespace check_perpendicularity_l508_508810

noncomputable def perpendicular_plane (a b α : Set) : Prop :=
  (∃ m n, m ≠ n ∧ b ∩ m ≠ ∅ ∧ b ∩ n ≠ ∅ ∧ m ⊆ α ∧ n ⊆ α) →
  (a ∩ m = ∅ ∧ a ∩ n = ∅)

noncomputable def lines_parallel (a b : Set) : Prop :=
  ∃ l, a ⊆ l ∧ b ⊆ l

theorem check_perpendicularity (a b α : Set) (h1 : lines_parallel a b)
  (h2 : perpendicular_plane b b α) : perpendicular_plane a b α :=
  sorry

end check_perpendicularity_l508_508810


namespace functions_with_inverses_l508_508665

noncomputable def a : ℝ → ℝ := λ x, real.sqrt (3 - x)
def a_domain : set ℝ := {x | x ≤ 3}

noncomputable def b : ℝ → ℝ := λ x, x^3 + x
def b_domain : set ℝ := set.univ

noncomputable def c : ℝ → ℝ := λ x, x - (1 / x)
def c_domain : set ℝ := {x | 0 < x}

noncomputable def d : ℝ → ℝ := λ x, 3 * x^2 + 6 * x + 10
def d_domain : set ℝ := {x | 0 ≤ x}

noncomputable def f : ℝ → ℝ := λ x, 2^x + 8^x
def f_domain : set ℝ := set.univ

noncomputable def h : ℝ → ℝ := λ x, x / 3
def h_domain : set ℝ := {x | -3 ≤ x ∧ x < 21}

theorem functions_with_inverses :
  ∃ (a_inv : {x : ℝ | x ≤ 3} → ℝ) (b_inv : ℝ → ℝ) (c_inv : {x : ℝ | 0 < x} → ℝ)
    (d_inv : {x : ℝ | 0 ≤ x} → ℝ) (f_inv : ℝ → ℝ) (h_inv : {x : ℝ | -3 ≤ x ∧ x < 21} → ℝ),
    (∀ x ∈ {x : ℝ | x ≤ 3}, a (a_inv x) = x ∧ a_inv (a x) = x) ∧
    (∀ x, b (b_inv x) = x ∧ b_inv (b x) = x) ∧
    (∀ x ∈ {x : ℝ | 0 < x}, c (c_inv x) = x ∧ c_inv (c x) = x) ∧
    (∀ x ∈ {x : ℝ | 0 ≤ x}, d (d_inv x) = x ∧ d_inv (d x) = x) ∧
    (∀ x, f (f_inv x) = x ∧ f_inv (f x) = x) ∧
    (∀ x ∈ {x : ℝ | -3 ≤ x ∧ x < 21}, h (h_inv x) = x ∧ h_inv (h x) = x) :=
sorry

end functions_with_inverses_l508_508665


namespace max_t_solution_set_l508_508441

noncomputable def f (x : ℝ) : ℝ := 1 / (9 * (Real.sin x)^2) + 4 / (9 * (Real.cos x)^2)

theorem max_t (x : ℝ) (hx : 0 < x ∧ x < Real.pi / 2) : ∃ t, ∀ x ∈ (Set.Ioo 0 (Real.pi / 2)), f x ≥ t ∧ t = 1 := 
sorry

theorem solution_set (x : ℝ) (hx : 0 < x ∧ x < Real.pi / 2) : (| x + 1 | + | x - 2 | ≥ 5) → (x ≤ -2 ∨ x ≥ 3) := 
sorry

end max_t_solution_set_l508_508441


namespace range_of_a_l508_508463

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, x^2 + a * x + 1 = 0) ∧ (∀ x : ℝ, exp (2 * x) - 2 * exp (x) + a ≥ 0) →
  a ≥ 0 :=
sorry

end range_of_a_l508_508463


namespace fourth_person_height_l508_508290

-- Definitions based on conditions
def h1 : ℕ := 73  -- height of first person
def h2 : ℕ := h1 + 2  -- height of second person
def h3 : ℕ := h2 + 2  -- height of third person
def h4 : ℕ := h3 + 6  -- height of fourth person

theorem fourth_person_height : h4 = 83 :=
by
  -- calculation to check the average height and arriving at h1
  -- (all detailed calculations are skipped using "sorry")
  sorry

end fourth_person_height_l508_508290


namespace least_possible_area_of_triangle_ABC_l508_508642

/--
Given a regular decagon formed by the complex solutions of the equation (z + 5)^10 = 32, 
with vertices of the decagon labeled as z_k = 2^(1/2) * (cos(2 * pi * k / 10) + i * sin(2 * pi * k / 10))
for k going from 0 to 9, prove the area of the triangle formed by three consecutive points is
(√50 - √10) / 8.
-/
theorem least_possible_area_of_triangle_ABC :
  let z (k : ℕ) := Complex.exp (Complex.I * (2 * Real.pi * k / 10)) * Real.sqrt 2
  in ∃ A B C : Complex, 
    ∃ k : ℕ, z k = A ∧ z (k + 1) % 10 = B ∧ z (k + 2) % 10 = C ∧
    abs ((B - A) * (C - A).conj) = (Real.sqrt 50 - Real.sqrt 10) / 8 :=
sorry

end least_possible_area_of_triangle_ABC_l508_508642


namespace num_common_tangents_l508_508460

noncomputable def circle1_center (a : ℝ) := (0, a)
noncomputable def circle1_radius (a : ℝ) := a
noncomputable def circle1 (a : ℝ) := { p : ℝ × ℝ | p.1^2 + (p.2 - a)^2 = a^2 }
noncomputable def circle2_center := (1, 2)
noncomputable def circle2_radius := 1
noncomputable def circle2 := { p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 2)^2 = 1 }

noncomputable def distance_to_line (x0 y0 : ℝ) := abs (x0 - y0 - 2) / real.sqrt (1^2 + 1^2)

theorem num_common_tangents (a : ℝ) (ha : a > 0) (hd : distance_to_line 0 a = 2 * real.sqrt 2) :
  let c1 := circle1_center a,
      r1 := circle1_radius a,
      c2 := circle2_center,
      r2 := circle2_radius,
      d := real.sqrt ((c2.1 - c1.1)^2 + (c2.2 - c1.2)^2)
  in d = abs (r1 - r2) → 
      -- Prove number of common tangents is 1
      1 :=
by
  sorry

end num_common_tangents_l508_508460


namespace smallest_k_inequality_l508_508338

theorem smallest_k_inequality : ∃ k : ℤ, (64^k > 4^16) ∧ (∀ m : ℤ, (64^m > 4^16) → k ≤ m) := 
sorry

end smallest_k_inequality_l508_508338


namespace ratio_flavoring_to_water_comparison_l508_508188

/-- The standard and sport formulation ratios -/
structure Formulation :=
  (flavoring : ℕ)
  (corn_syrup : ℕ)
  (water : ℕ)

def standard_formulation := Formulation.mk 1 12 30

def sport_corn_syrup (standard_ratio : Formulation) := 3 * standard_ratio.flavoring

def sport_formulation (cs_flavoring_ratio : ℕ) (corn_syrup_amt water_amt : ℕ) :=
  Formulation.mk (corn_syrup_amt / cs_flavoring_ratio)  corn_syrup_amt water_amt

theorem ratio_flavoring_to_water_comparison :
  let standard_sport_corn_syrup_ratio := sport_corn_syrup standard_formulation
  ∃ (sport : Formulation), 
    sport.corn_syrup = 7 
    ∧ sport.water = 105 
    ∧ let sport_ratio := sport.flavoring / sport.water
       (sport_ratio : ℚ) = (1 : ℚ) / (2 : ℚ) * ((standard_formulation.flavoring / standard_formulation.water : ℚ))  :=
by 
  sorry

end ratio_flavoring_to_water_comparison_l508_508188


namespace total_cases_sold_l508_508300

theorem total_cases_sold : 
  let people := 20 in
  let first_8_cases := 8 * 3 in
  let next_4_cases := 4 * 2 in
  let last_8_cases := 8 * 1 in
  first_8_cases + next_4_cases + last_8_cases = 40 := 
by
  let people := 20
  let first_8_cases := 8 * 3
  let next_4_cases := 4 * 2
  let last_8_cases := 8 * 1
  have h1 : first_8_cases = 24 := by rfl
  have h2 : next_4_cases = 8 := by rfl
  have h3 : last_8_cases = 8 := by rfl
  have h : first_8_cases + next_4_cases + last_8_cases = 24 + 8 + 8 := by rw [h1, h2, h3]
  show 24 + 8 + 8 = 40 from rfl

end total_cases_sold_l508_508300


namespace comb_1_a_comb_1_b_comb_2_l508_508576

theorem comb_1_a (n k : ℕ) (hn : n ≥ 3) (hk : k ≥ 1) :
  k * Nat.choose n k - n * Nat.choose (n - 1) (k - 1) = 0 := sorry

theorem comb_1_b (n k : ℕ) (hn : n ≥ 3) (hk : k ≥ 2) :
  k^2 * Nat.choose n k - n * (n - 1) * Nat.choose (n - 2) (k - 2) - n * Nat.choose (n - 1) (k - 1) = 0 := sorry

theorem comb_2 (n : ℕ) (hn : n ≥ 3) :
  (Finset.sum (Finset.range (n+1)) (λ k, (k+1)^2 * Nat.choose n k)) = 2^(n-2) * (n^2 + 5 * n + 4) := sorry

end comb_1_a_comb_1_b_comb_2_l508_508576


namespace unique_solution_c_value_l508_508081

-- Define the main problem: the parameter c for which a given system of equations has a unique solution.
theorem unique_solution_c_value (c : ℝ) : 
  (∀ x y : ℝ, 2 * abs (x + 7) + abs (y - 4) = c ∧ abs (x + 4) + 2 * abs (y - 7) = c → 
   (x = -7 ∧ y = 7)) ↔ c = 3 :=
by sorry

end unique_solution_c_value_l508_508081


namespace savings_after_increase_l508_508003

/-- A man saves 20% of his monthly salary. If on account of dearness of things
    he is to increase his monthly expenses by 20%, he is only able to save a
    certain amount per month. His monthly salary is Rs. 6250. -/
theorem savings_after_increase (monthly_salary : ℝ) (initial_savings_percentage : ℝ)
  (increase_expenses_percentage : ℝ) (final_savings : ℝ) :
  monthly_salary = 6250 ∧
  initial_savings_percentage = 0.20 ∧
  increase_expenses_percentage = 0.20 →
  final_savings = 250 :=
by
  sorry

end savings_after_increase_l508_508003


namespace scientific_notation_11580000_l508_508916

theorem scientific_notation_11580000 :
  11580000 = 1.158 * 10^7 :=
sorry

end scientific_notation_11580000_l508_508916


namespace seating_arrangements_l508_508955

theorem seating_arrangements (Yi Bing Ding Wu Jia : Type) [DecidableEq Type] :
  let seating_arrangements := 12 in
  seating_arrangements = 12 :=
by
  sorry

end seating_arrangements_l508_508955


namespace prob1_prob2_prob3_prob4_prob5_l508_508679

theorem prob1 : (1 - 27 + (-32) + (-8) + 27) = -40 := sorry

theorem prob2 : (2 * -5 + abs (-3)) = -2 := sorry

theorem prob3 (x y : Int) (h₁ : -x = 3) (h₂ : abs y = 5) : x + y = 2 ∨ x + y = -8 := sorry

theorem prob4 : ((-1 : Int) * (3 / 2) + (5 / 4) + (-5 / 2) - (-13 / 4) - (5 / 4)) = -3 / 4 := sorry

theorem prob5 (a b : Int) (h : abs (a - 4) + abs (b + 5) = 0) : a - b = 9 := sorry

end prob1_prob2_prob3_prob4_prob5_l508_508679


namespace train_speed_l508_508379

theorem train_speed (d : ℝ) (t : ℝ) (speed : ℝ) :
  d = 140 ∧ t = 2.3998080153587713 ∧ speed = (d / t) * 3.6 → speed ≈ 210.0348 :=
by
  intro h
  cases h with h1 h_rest
  cases h_rest with h2 h3
  rw h1 at h3
  rw h2 at h3
  sorry

end train_speed_l508_508379


namespace ratio_of_segments_l508_508340

open EuclideanGeometry

variable (A B C D M E F : Point)
variable (h_parallelogram : parallelogram A B C D)
variable (h_M_on_AC : collinear_points A C M)
variable (h_ME_perp_AB : orthogonal_perp_segment M E A B)
variable (h_MF_perp_AD : orthogonal_perp_segment M F A D)

theorem ratio_of_segments (h_parallelogram : parallelogram A B C D) 
                          (h_M_on_AC : collinear_points A C M) 
                          (h_ME_perp_AB : orthogonal_perp_segment M E A B) 
                          (h_MF_perp_AD : orthogonal_perp_segment M F A D) :
  ME / MF = AD / AB := 
by 
  sorry

end ratio_of_segments_l508_508340


namespace find_sum_p_q_l508_508186

-- Definitions of the given lengths and equalities
variables (A B C D E : Type) [E : EuclideanGeometry]
variables (AB BD BC AD : Real)
variables (p q : Nat)

-- Given conditions
axiom angle_BAD_cong_angle_CBD : E.angle BAD = E.angle CBD
axiom angle_ADB_cong_angle_BCD : E.angle ADB = E.angle BCD
axiom AB_equals_7 : AB = 7
axiom BD_equals_11 : BD = 11
axiom BC_equals_9 : BC = 9
axiom AD_equals_121_div_7 : AD = (121 / 7)

-- The theorem we need to prove
theorem find_sum_p_q : p = 121 → q = 7 → p + q = 128 :=
by
  intro p_eq q_eq
  simp [p_eq, q_eq]
  rfl

end find_sum_p_q_l508_508186


namespace fractional_part_sum_l508_508573

theorem fractional_part_sum {n a b : ℤ} (hn : n > 0) (hp : Nat.gcd a n = 1) :
  (Finset.sum (Finset.range n) (λ k, fract ((k * a + b) / n))) = (n - 1) / 2 :=
by
  sorry

end fractional_part_sum_l508_508573


namespace find_A_for_club_suit_l508_508160

def club_suit (A B : ℝ) : ℝ := 3 * A + 2 * B^2 + 5

theorem find_A_for_club_suit :
  ∃ A : ℝ, club_suit A 3 = 73 ∧ A = 50 / 3 :=
sorry

end find_A_for_club_suit_l508_508160


namespace kanul_spent_on_machinery_l508_508555

-- Definitions
variable R : ℝ := 3000
variable T : ℝ := 5555.56
variable C : ℝ := 0.10 * T
variable M : ℝ 

-- The main theorem to be proved
theorem kanul_spent_on_machinery : 
  R + M + C = T → M = 2000 := 
by
  intro h
  sorry

end kanul_spent_on_machinery_l508_508555


namespace anil_multiplication_problem_l508_508333

theorem anil_multiplication_problem (x : ℕ) (h : 53 * x - 35 * x = 540) : 53 * x = 1590 :=
by
  have hx : x = 30 := by
    calc
      x = 540 / 18 := by rw [Nat.div_eq_of_eq_mul_left (Nat.zero_lt_succ _) rfl, Nat.div_self (Nat.pos_of_ne_zero _), rfl]
      .. by rw [h, ← Nat.mul_sub_left_distrib, add_right_neg, Nat.mul_comm, add_zero]
  rw [hx, Nat.mul_comm]
  rfl

end anil_multiplication_problem_l508_508333


namespace melanie_mother_dimes_l508_508992

-- Definitions based on the conditions
variables (initial_dimes : ℕ) (dimes_given_to_dad : ℤ) (current_dimes : ℤ)

-- Conditions
def melanie_conditions := initial_dimes = 7 ∧ dimes_given_to_dad = 8 ∧ current_dimes = 3

-- Question to be proved is equivalent to proving the number of dimes given by her mother
theorem melanie_mother_dimes (initial_dimes : ℕ) (dimes_given_to_dad : ℤ) (current_dimes : ℤ) (dimes_given_by_mother : ℤ) 
  (h : melanie_conditions initial_dimes dimes_given_to_dad current_dimes) : 
  dimes_given_by_mother = 4 :=
by 
  sorry

end melanie_mother_dimes_l508_508992


namespace line_through_midpoints_parallel_and_bisects_perimeter_l508_508192

/-- In triangle ABC, AC and BC sides are tangent to circles at points K and L respectively. 
Prove that the line passing through the midpoints of segments KL and AB is
parallel to the angle bisector of ∠ ACB and bisects the perimeter of the triangle. -/
theorem line_through_midpoints_parallel_and_bisects_perimeter
  {A B C K L : Point}
  (h_tangent_A : ∃ r, circle A r ∩ line_segment A C = {K})
  (h_tangent_B : ∃ r, circle B r ∩ line_segment B C = {L})
  (h_triangle : Triangle A B C) :
  ∃ M N : Point,
  midpoint K L = M ∧
  midpoint A B = N ∧
  (is_parallel (line_through M N) (angle_bisector A C B)) ∧
  (perimeter_bisected (line_through M N) (Triangle A B C)) :=
sorry

end line_through_midpoints_parallel_and_bisects_perimeter_l508_508192


namespace distance_MN_zero_l508_508206

-- Definitions for the points of rectangle
def A : (ℝ × ℝ × ℝ) := (0, 0, 0)
def B : (ℝ × ℝ × ℝ) := (2, 0, 0)
def C : (ℝ × ℝ × ℝ) := (2, 3, 0)
def D : (ℝ × ℝ × ℝ) := (0, 3, 0)

-- Definitions for new points along rays
-- Coordinates given directly
def A' : (ℝ × ℝ × ℝ) := (0, 1, 12)
def B' : (ℝ × ℝ × ℝ) := (2, 0, 14)
def C' : (ℝ × ℝ × ℝ) := (2, 2, 20)
def D' : (ℝ × ℝ × ℝ) := (0, 3, 18)

-- Midpoints of segments A'C' and B'D'
def M : (ℝ × ℝ × ℝ) := ((A'.1 + C'.1) / 2, (A'.2 + C'.2) / 2, (A'.3 + C'.3) / 2)
def N : (ℝ × ℝ × ℝ) := ((B'.1 + D'.1) / 2, (B'.2 + D'.2) / 2, (B'.3 + D'.3) / 2)

-- Statement to prove: distance between midpoints M and N is 0
theorem distance_MN_zero : (M.1 - N.1)^2 + (M.2 - N.2)^2 + (M.3 - N.3)^2 = 0 := by sorry

end distance_MN_zero_l508_508206


namespace find_log_b_tan_x_l508_508156

noncomputable def log_b_tan_x (b x a : ℝ) (hb : b > 1) (h1 : sin x > 0) (h2 : cos x > 0) (h3 : log b (sin x) = a) : Prop :=
  log b (tan x) = a - (1 / 2) * log b (1 - b^(2 * a))

theorem find_log_b_tan_x (b x a : ℝ) (hb : b > 1) (h1 : sin x > 0) (h2 : cos x > 0) (h3 : log b (sin x) = a) : 
  log b (tan x) = a - (1 / 2) * log b (1 - b^(2 * a)) := 
by 
  sorry

end find_log_b_tan_x_l508_508156


namespace second_shift_fraction_of_total_l508_508334

theorem second_shift_fraction_of_total (W E : ℕ) (h1 : ∀ (W : ℕ), E = (3 * W / 4))
  : let W₁ := W
    let E₁ := E
    let widgets_first_shift := W₁ * E₁
    let widgets_per_second_shift_employee := (2 * W₁) / 3
    let second_shift_employees := (4 * E₁) / 3
    let widgets_second_shift := (2 * W₁ / 3) * (4 * E₁ / 3)
    let total_widgets := widgets_first_shift + widgets_second_shift
    let fraction_second_shift := widgets_second_shift / total_widgets
    fraction_second_shift = 8 / 17 :=
sorry

end second_shift_fraction_of_total_l508_508334


namespace dot_product_ab_angle_ab_l508_508975

def vec_a : ℝ × ℝ := (3, 4)
def vec_b : ℝ × ℝ := (-1, 7)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 * v.1 + v.2 * v.2)

def cos_angle (u v : ℝ × ℝ) : ℝ :=
  (dot_product u v) / ((magnitude u) * (magnitude v))

theorem dot_product_ab : dot_product vec_a vec_b = 25 := by
  sorry

theorem angle_ab : Real.arccos (cos_angle vec_a vec_b) = Real.pi / 4 := by
  sorry

end dot_product_ab_angle_ab_l508_508975


namespace vector_OC_equation_l508_508829

variables {V : Type*} [AddCommGroup V] [Module ℝ V] 
variable (O A B C : V)
variable (h1 : ﻿2 • (B - A) + (B - C) = 0)

theorem vector_OC_equation : (C - O) = -2 • (A - O) + 3 • (B - O) :=
by
  sorry

end vector_OC_equation_l508_508829


namespace conference_distance_l508_508255

-- Definitions for given conditions
def time_hours (t : ℝ) : Prop := t = 0.25
def distance_40 (d : ℝ) : Prop := d = 40 * (time_hours t + 0.75)
def speed_increase_60 (d : ℝ) : Prop := d - 40 = 60 * (time_hours t - 0.25)

-- The theorem to prove the distance is 40 miles
theorem conference_distance (t d : ℝ) (dist40 : distance_40 d) (speed60 : speed_increase_60 d) : d = 40 :=
by sorry

end conference_distance_l508_508255


namespace profit_share_difference_l508_508370

noncomputable def value_of_each_part (total_parts value_per_part : ℝ) : ℝ :=
total_parts * value_per_part

noncomputable def difference (highest-share lowest-share : ℝ) : ℝ :=
highest-share - lowest-share

theorem profit_share_difference : 
  let total_profit : ℝ := 700
  let ratio_x : ℝ := 1 / 3
  let ratio_y : ℝ := 1 / 4
  let ratio_z : ℝ := 1 / 5
  let total_parts := ratio_x + ratio_y + ratio_z
  let value_per_part := total_profit * (60 / total_parts)
  let x_share := value_of_each_part (ratio_x * 60) value_per_part
  let z_share := value_of_each_part (ratio_z * 60) value_per_part
  in difference x_share z_share ≈ 7148.93 :=
by
  sorry

end profit_share_difference_l508_508370


namespace surface_area_of_modified_octahedron_volume_of_modified_octahedron_l508_508312

theorem surface_area_of_modified_octahedron (a : ℝ) : 
  F = π * (4 * sqrt 6 - 7) / 3 * (a ^ 2) := sorry

theorem volume_of_modified_octahedron (a : ℝ) : 
  V = π * (14 * sqrt 6 - 27) / 54 * (a ^ 3) := sorry

end surface_area_of_modified_octahedron_volume_of_modified_octahedron_l508_508312


namespace sum_of_first_twelve_terms_l508_508432

-- Define the arithmetic sequence with given conditions
variable {a d : ℚ}

-- The fifth term of the sequence
def a5 : ℚ := a + 4 * d

-- The seventeenth term of the sequence
def a17 : ℚ := a + 16 * d

-- Sum of the first twelve terms of the arithmetic sequence
def S12 (a d : ℚ) : ℚ := 6 * (2 * a + 11 * d)

theorem sum_of_first_twelve_terms (a : ℚ) (d : ℚ) (h₁ : a5 = 1) (h₂ : a17 = 18) :
  S12 a d = 37.5 := by
  sorry

end sum_of_first_twelve_terms_l508_508432


namespace percentage_error_in_side_l508_508032

theorem percentage_error_in_side {S S' : ℝ}
  (hs : S > 0)
  (hs' : S' > S)
  (h_area_error : (S'^2 - S^2) / S^2 * 100 = 90.44) :
  ((S' - S) / S * 100) = 38 :=
by
  sorry

end percentage_error_in_side_l508_508032


namespace day_of_300th_day_of_previous_year_is_monday_l508_508517

-- Define a function that converts a day of the week number (1=Monday, 2=Tuesday, ..., 7=Sunday)
-- to the previous day of the week.
def prev_day (day : ℕ) : ℕ :=
  if day = 1 then 7 else day - 1

-- Define a function that calculates the day of the week after moving n days back from a given day.
def days_back (start_day : ℕ) (n : ℕ) : ℕ :=
  nat.rec_on n start_day (λ _ prev, prev_day prev)

theorem day_of_300th_day_of_previous_year_is_monday
  (currentYear : ℕ)
  (nextYear : ℕ)
  (previousYear : ℕ)
  (day_of_week : ℕ → ℕ)
  (sunday : ℕ)
  (h1 : day_of_week (currentYear * 365 + 200) = sunday)
  (h2 : day_of_week ((currentYear + 1) * 365 + 100) = sunday) :
  day_of_week ((currentYear - 1) * 365 + 300) = 1 :=
by
  sorry

end day_of_300th_day_of_previous_year_is_monday_l508_508517


namespace stratified_sampling_grade10_l508_508018

theorem stratified_sampling_grade10
  (total_students : ℕ)
  (grade10_students : ℕ)
  (grade11_students : ℕ)
  (grade12_students : ℕ)
  (sample_size : ℕ)
  (h1 : total_students = 700)
  (h2 : grade10_students = 300)
  (h3 : grade11_students = 200)
  (h4 : grade12_students = 200)
  (h5 : sample_size = 35)
  : (grade10_students * sample_size / total_students) = 15 := 
sorry

end stratified_sampling_grade10_l508_508018


namespace brick_height_calculation_l508_508304

theorem brick_height_calculation :
  ∀ (num_bricks : ℕ) (brick_length brick_width brick_height : ℝ)
    (wall_length wall_height wall_width : ℝ),
    num_bricks = 1600 →
    brick_length = 100 →
    brick_width = 11.25 →
    wall_length = 800 →
    wall_height = 600 →
    wall_width = 22.5 →
    wall_length * wall_height * wall_width = 
    num_bricks * brick_length * brick_width * brick_height →
    brick_height = 60 :=
by
  sorry

end brick_height_calculation_l508_508304


namespace range_of_m_l508_508170

def parabola_meets_segment (y : ℝ → ℝ) (l : ℝ → ℝ) (m : ℝ) : Prop :=
  let f := λ x : ℝ, x^2 - m * x + m + 1
  f = y ∧ l = λ x, -x + 4 ∧ ∃ x1 x2 : ℝ, 0 ≤ x1 ∧ x1 < x2 ∧ x2 ≤ 4 ∧ f x1 = l x1 ∧ f x2 = l x2

theorem range_of_m (m : ℝ) :
  (3 ≤ m ∧ m ≤ 17 / 3) ↔ parabola_meets_segment (λ x, x^2 - m * x + m + 1) (λ x, -x + 4) m :=
by {
  sorry
}

end range_of_m_l508_508170


namespace number_of_boys_l508_508614

-- Definitions from the problem conditions
def trees : ℕ := 29
def trees_per_boy : ℕ := 3

-- Prove the number of boys is 10
theorem number_of_boys : (trees / trees_per_boy) + 1 = 10 :=
by sorry

end number_of_boys_l508_508614


namespace triangle_GHX_area_l508_508262

variable (HEXAGN : Type) [Hexagon HEXAGN]
variable (P : Point)
variable (X G H : Point)

-- Assumptions from the problem
axiom hexagon_rotational_symmetry (h : HEXAGN → Prop) (P : Point) : 
  ∀ hx, h hx → rotated P 120 hx = hx
axiom PX_equal_1 : distance P X = 1

-- Target: Prove that the area of triangle GHX is sqrt(3)/4 given the conditions
theorem triangle_GHX_area :
  ∀ (HEXAGN : Type) [Hexagon HEXAGN] (P X G H : Point),
  (hexagon_rotational_symmetry HEXAGN P) →
  (PX_equal_1 P X) →
  area_of_triangle G H X = (sqrt 3) / 4 :=
by
  sorry

end triangle_GHX_area_l508_508262


namespace find_bisecting_line_l508_508461

-- Definitions based on conditions
def P := (0 : ℝ, 1 : ℝ)
def l1 := {p : ℝ × ℝ | 2 * p.1 + p.2 - 8 = 0}
def l2 := {p : ℝ × ℝ | p.1 - 3 * p.2 + 10 = 0}

-- Main statement to prove
theorem find_bisecting_line :
  ∃ l : set (ℝ × ℝ), 
    (∀ p : ℝ × ℝ, p ∈ l ↔ p.1 + 4 * p.2 - 4 = 0) ∧
    ∀ a : ℝ, (a, 8 - 2 * a) ∈ l1 →
              (⟨-a, -2 * a - 6⟩ ∈ l2 → 
               (2 * a + (-2 * a - 4) ∈ l)) :=
sorry

end find_bisecting_line_l508_508461


namespace solve_expression_l508_508848

-- Define the relevant expressions
def term1 : ℝ := (- (7/8))^0
def term2 : ℝ := [(-2)^3]^( - (2/3))

-- State the theorem
theorem solve_expression : term1 + term2 = 5/4 :=
by sorry

end solve_expression_l508_508848


namespace proof_problem_l508_508945

noncomputable def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ := (ρ * cos θ, ρ * sin θ)

noncomputable def circle_eq (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = 8

noncomputable def line_eq (x y t : ℝ) : Prop := x = -2 + t/2 ∧ y = -3 + t * (Real.sqrt 3) / 2

noncomputable def line_cartesian (x y : ℝ) : Prop := Real.sqrt 3 * x - y + 2 * Real.sqrt 3 - 3 = 0

theorem proof_problem (ρ θ t : ℝ) (hρ : ρ = 4 * Real.sqrt 2 * sin (θ + π/4))
  (hx : x = -2 + t/2) (hy : y = -3 + t * (Real.sqrt 3) / 2) :
  (x, y) ∈ line_cartesian → circle_eq x y → 
  |PA| * |PB| = 33 :=
by
  sorry

end proof_problem_l508_508945


namespace intersection_M_complement_N_eq_l508_508139

open Set

noncomputable def U : Set ℝ := univ
noncomputable def M : Set ℝ := {x | x^2 - 2 * x - 3 ≤ 0}
noncomputable def N : Set ℝ := {y | ∃ x : ℝ, y = x^2 + 1}
noncomputable def complement_N : Set ℝ := {y | y < 1}

theorem intersection_M_complement_N_eq : M ∩ complement_N = {x | -1 ≤ x ∧ x < 1} :=
by
  sorry

end intersection_M_complement_N_eq_l508_508139


namespace males_in_sample_l508_508354

theorem males_in_sample (total_employees female_employees sample_size : ℕ) 
  (h1 : total_employees = 300)
  (h2 : female_employees = 160)
  (h3 : sample_size = 15)
  (h4 : (female_employees * sample_size) / total_employees = 8) :
  sample_size - ((female_employees * sample_size) / total_employees) = 7 :=
by
  sorry

end males_in_sample_l508_508354


namespace min_cos_C_geq_one_third_l508_508909

variable (A B C : ℝ) (a b c : ℝ)
variable (h1 : ∀ {A B C : ℕ}, 1 / tan A + 1 / tan B = 4 / tan C)
variable [A + B + C = π]

theorem min_cos_C_geq_one_third 
  (h1 : 1 / tan A + 1 / tan B = 4 / tan C) 
  (h2 : A + B + C = π) 
  : cos C ≥ 1 / 3 := 
  sorry

end min_cos_C_geq_one_third_l508_508909


namespace product_floor_ceil_sequence_l508_508412

noncomputable def floor (x : ℝ) : ℤ := Int.floor x
noncomputable def ceil (x : ℝ) : ℤ := Int.ceil x

theorem product_floor_ceil_sequence :
    (floor (-6 - 0.5) * ceil (6 + 0.5)) *
    (floor (-5 - 0.5) * ceil (5 + 0.5)) *
    (floor (-4 - 0.5) * ceil (4 + 0.5)) *
    (floor (-3 - 0.5) * ceil (3 + 0.5)) *
    (floor (-2 - 0.5) * ceil (2 + 0.5)) *
    (floor (-1 - 0.5) * ceil (1 + 0.5)) *
    (floor (-0.5) * ceil (0.5)) = -25401600 :=
by
  sorry

end product_floor_ceil_sequence_l508_508412


namespace find_numbers_between_70_and_80_with_gcd_6_l508_508074

theorem find_numbers_between_70_and_80_with_gcd_6 :
  ∃ n, 70 ≤ n ∧ n ≤ 80 ∧ Nat.gcd n 30 = 6 ∧ (n = 72 ∨ n = 78) :=
by
  sorry

end find_numbers_between_70_and_80_with_gcd_6_l508_508074


namespace find_x_l508_508601

-- Declare points A, B, C, D lying on the same line, and point E not on the line
-- with specific distances given between them and the condition on the perimeter.
theorem find_x (A B C D E : Point) (x : ℝ)
  (hAB : dist A B = x) (hCD : dist C D = x) (hBC : dist B C = 18)
  (hBE : dist B E = 10) (hCE : dist C E = 18) :
  3 * (dist B E + dist E C + dist B C) = dist A E + dist E D + dist A D →
  x = 9 :=
  sorry

end find_x_l508_508601


namespace triangle_BD_length_l508_508190

noncomputable def length_BD (AB AC BC : ℝ) (is_right_angle : angle_at_right AB AC BC) (BD AD : ℝ) : Prop :=
BD = 56.25

theorem triangle_BD_length (AB AC BC : ℝ) (is_right_angle : angle_at_right AB AC BC) (BD AD : ℝ) 
(h1 : AB = 45)
(h2 : AC = 60)
(h3 : ∃ D : point, lies_on_line D BC ∧ AD ⊥ BC)
:
length_BD AB AC BC is_right_angle BD AD :=
by
  sorry

end triangle_BD_length_l508_508190


namespace pastries_made_initially_l508_508387

theorem pastries_made_initially 
  (sold : ℕ) (remaining : ℕ) (initial : ℕ) 
  (h1 : sold = 103) (h2 : remaining = 45) : 
  initial = 148 :=
by
  have h := h1
  have r := h2
  sorry

end pastries_made_initially_l508_508387


namespace all_tickets_can_be_placed_in_50_boxes_not_possible_with_less_than_40_boxes_not_possible_with_less_than_50_boxes_all_four_digit_tickets_can_be_placed_in_34_boxes_minimal_boxes_for_k_digit_tickets_l508_508293

-- Part (a) 
theorem all_tickets_can_be_placed_in_50_boxes :
  ∀ (tickets : list (fin 1000)) (boxes : list (fin 100)), 
  (∀ t ∈ tickets, ∃ b ∈ boxes, 
    (∃ i, remove_i_digit t i = b)) → 
  (∃ boxes50 : list (fin 100), boxes50.length = 50 ∧ 
    ∀ t ∈ tickets, ∃ b ∈ boxes50, 
    (∃ i, remove_i_digit t i = b)) :=
by sorry

-- Part (b) 
theorem not_possible_with_less_than_40_boxes :
  ∀ (tickets : list (fin 1000)), 
  ¬ ∃ (boxes : list (fin 100)), boxes.length < 40 ∧ 
  (∀ t ∈ tickets, ∃ b ∈ boxes, 
    (∃ i, remove_i_digit t i = b)) :=
by sorry

-- Part (c) 
theorem not_possible_with_less_than_50_boxes :
  ∀ (tickets : list (fin 1000)), 
  ¬ ∃ (boxes : list (fin 100)), boxes.length < 50 ∧ 
  (∀ t ∈ tickets, ∃ b ∈ boxes, 
    (∃ i, remove_i_digit t i = b)) :=
by sorry

-- Part (d) 
theorem all_four_digit_tickets_can_be_placed_in_34_boxes :
  ∀ (tickets : list (fin 10000)) (boxes : list (fin 100)), 
  (∀ t ∈ tickets, ∃ b ∈ boxes, 
    (∃ i j, remove_ij_digits t i j = b)) → 
  (∃ boxes34 : list (fin 100), boxes34.length = 34 ∧ 
    ∀ t ∈ tickets, ∃ b ∈ boxes34, 
    (∃ i j, remove_ij_digits t i j = b)) :=
by sorry

-- Part (e) 
theorem minimal_boxes_for_k_digit_tickets :
  ∀ (k : ℕ) (tickets : list (fin (10^k))) (boxes : list (fin (10^(k-2)))), 
  (∀ t ∈ tickets, ∃ b ∈ boxes, 
    (∃ indices : list nat, indices.length = 2 ∧ remove_indices t indices = b)) → 
  (∃ min_boxes : ℕ, (∀ boxes' : list (fin (10^(k-2))), 
    boxes'.length < min_boxes → 
    ¬ ∀ t ∈ tickets, ∃ b ∈ boxes', 
    (∃ indices : list nat, indices.length = 2 ∧ remove_indices t indices = b))) :=
by sorry

end all_tickets_can_be_placed_in_50_boxes_not_possible_with_less_than_40_boxes_not_possible_with_less_than_50_boxes_all_four_digit_tickets_can_be_placed_in_34_boxes_minimal_boxes_for_k_digit_tickets_l508_508293


namespace number_of_valid_positions_l508_508309

-- Definitions for truth tellers and liars
inductive Person
| truth_teller (n : ℕ)
| liar (n : ℕ)

open Person

-- Let there be exactly two truth tellers and two liars
def is_truth_teller : Person → Prop
| (truth_teller _) := true
| _ := false

def is_liar : Person → Prop
| (liar _) := true
| _ := false

-- Condition that everyone must claim that all people directly adjacent to them are liars
def adjacent_claim (p1 p2 : Person) : Prop :=
is_liar p1 ∧ is_truth_teller p2 ∨ 
is_truth_teller p1 ∧ is_liar p2

def all_adjacent_claims (l : List Person) : Prop :=
match l with
| []           => true
| [x]          => true
| x :: y :: xs => adjacent_claim x y ∧ all_adjacent_claims (y :: xs)
end

-- Main theorem to prove
theorem number_of_valid_positions : 
  ∃ l : List Person, 
    all_adjacent_claims l ∧ 
    l.length = 4 ∧ 
    l.countp is_truth_teller = 2 ∧ 
    l.countp is_liar = 2 ∧
    (List.permutations [truth_teller 1, truth_teller 2, liar 1, liar 2]).count all_adjacent_claims = 8 :=
sorry

end number_of_valid_positions_l508_508309


namespace problem_statement_l508_508577

theorem problem_statement :
  ∃ (w x y z : ℕ), (2^w * 3^x * 5^y * 7^z = 588) ∧ (2 * w + 3 * x + 5 * y + 7 * z = 21) :=
by
  sorry

end problem_statement_l508_508577


namespace problem2024_l508_508055

theorem problem2024 :
  (∑ k in Finset.range 2023 | (2024 - k) / (k + 1)) / (∑ k in (Finset.range 2023) + 1 / (k + 2)) = 2024 := sorry

end problem2024_l508_508055


namespace sphere_volume_l508_508473

noncomputable def volume_of_the_sphere (A B C D : Point) (r : ℝ) : ℝ :=
  4 / 3 * Real.pi * r ^ 3

theorem sphere_volume :
  ∃ (A B C D : Point),
    (dist A B = 1) ∧
    (dist B C = 1) ∧
    (dist A D = 1) ∧
    (dist A C = Real.sqrt 2) ∧
    (dist B D = Real.sqrt 2) ∧
    (orthogonal A C B D) ∧
    (volume_of_the_sphere A B C D (Real.sqrt 3 / 2) = (Real.sqrt 3 / 2) * Real.pi) :=
by
  sorry

end sphere_volume_l508_508473


namespace lucas_raspberry_candies_l508_508989

-- Define the problem conditions and the question
theorem lucas_raspberry_candies :
  ∃ (r l : ℕ), (r = 3 * l) ∧ ((r - 5) = 4 * (l - 5)) ∧ (r = 45) :=
by
  sorry

end lucas_raspberry_candies_l508_508989


namespace domain_of_function_l508_508276

theorem domain_of_function :
  (∀ x : ℝ, f x = sqrt (3 - x^2) + 3 / (x - 1) →
    (3 - x^2 ≥ 0) ∧ (x - 1 ≠ 0)) →
  ∀ x : ℝ, (x ∈ Set.Icc (-(Real.sqrt 3)) (Real.sqrt 3) ∧ x ≠ 1) ↔ 
      (x ∈ Set.Icc (-(Real.sqrt 3)) (1- 1) ∪ (1+1) (Real.sqrt 3)) :=
sorry

end domain_of_function_l508_508276


namespace bernoulli_convergence_l508_508224

noncomputable theory

variables {Ω : Type*} [MeasureTheory.ProbabilitySpace Ω]

-- Definition of Bernoulli random variables
def bernoulli (p : ℝ) [hp : 0 < p ∧ p < 1] (n : ℕ) : Ω →₀ ℝ := {
  to_fun := λ ω, if random_var ω < p then 1 else 0,
  measurable' := measurable_of_measurable_coe (measurable_set_lt measurable_random_var (measurable_const p).measurable)
}

-- Condition: Sequence of independent Bernoulli random variables 
def bernoulli_sequence (p : ℝ) [hp : 0 < p ∧ p < 1] : ℕ → Ω →₀ ℝ :=
λ n, bernoulli p n

-- Condition: Sum of series
def sum_series (p : ℝ) [hp : 0 < p ∧ p < 1] := 
  ∑' (n : ℕ), (λ ω, (bernoulli_sequence p n ω) / 2^(n+1))

-- Theorem statement
theorem bernoulli_convergence (p : ℝ) [hp : 0 < p ∧ p < 1] :
  ∃ U, 
    (if p = 1/2 
      then measure_theory.probability_measure U = measure_theory.uniform (0:ℝ, 1:ℝ)
      else is_singular U) :=
begin
  sorry,
end

end bernoulli_convergence_l508_508224


namespace smallest_even_abundant_gt_12_l508_508703

def is_abundant (n : ℕ) : Prop :=
  (∑ d in (finset.filter (λ d, d < n ∧ n % d = 0) (finset.range n)), d) > n

theorem smallest_even_abundant_gt_12 : ∃ n : ℕ, n > 12 ∧ even n ∧ is_abundant n ∧ (∀ m : ℕ, m > 12 ∧ even m ∧ is_abundant m → n ≤ m) := 
by
  use 18
  split
  { norm_num }
  split
  { norm_num }
  split
  { unfold is_abundant
    norm_num }
  { intros m h
    sorry }

end smallest_even_abundant_gt_12_l508_508703


namespace total_biscuits_needed_l508_508241

-- Definitions
def number_of_dogs : ℕ := 2
def biscuits_per_dog : ℕ := 3

-- Theorem statement
theorem total_biscuits_needed : number_of_dogs * biscuits_per_dog = 6 :=
by sorry

end total_biscuits_needed_l508_508241


namespace sum_of_4_elem_subsets_is_11901_l508_508400

def num_of_4_elem_subsets : ℕ := (
  let A_0 := {3, 6, 9, 12, 15, 18};
  let A_1 := {1, 4, 7, 10, 13, 16, 19};
  let A_2 := {2, 5, 8, 11, 14, 17, 20};
  let comb := λ n k, (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)));
  (comb 7 2) * (comb 7 2) + 
  2 * (comb 6 1) * (comb 7 3) + 
  (comb 6 2) * 7 * 7 +
  (comb 6 4)
)

theorem sum_of_4_elem_subsets_is_11901 :
  num_of_4_elem_subsets = 11901 := by
  sorry

end sum_of_4_elem_subsets_is_11901_l508_508400


namespace trig_identity_proof_l508_508471

theorem trig_identity_proof
  (A B C : ℝ)
  (h1 : sin A + sin B + sin C = 0)
  (h2 : cos A + cos B + cos C = 0) :
  cos (3 * A) + cos (3 * B) + cos (3 * C) = 3 * cos (A + B + C)
  ∧ sin (3 * A) + sin (3 * B) + sin (3 * C) = 3 * sin (A + B + C) := 
by {
  sorry
}

end trig_identity_proof_l508_508471


namespace complex_prod_l508_508187

def z : ℂ := 1 + 2 * complex.i

theorem complex_prod (z : ℂ) (h : z = 1 + 2 * complex.i) : z * complex.i = -2 + complex.i := by
  rw [h]
  simp [mul_assoc, complex.i_mul_i]
  simp
  congr
  ring
sorry

end complex_prod_l508_508187


namespace probability_colors_match_l508_508720

noncomputable def alice_jellybean_prob {color : Type} (alice_jellybeans : color → ℕ) (picked_color : color) : ℚ :=
  let total_jellybeans := ∑ c, alice_jellybeans c
  (alice_jellybeans picked_color) / total_jellybeans

noncomputable def carl_jellybean_prob {color : Type} (carl_jellybeans : color → ℕ) (picked_color : color) : ℚ :=
  let total_jellybeans := ∑ c, carl_jellybeans c
  (carl_jellybeans picked_color) / total_jellybeans

theorem probability_colors_match :
  ∀ (color : Type)
    (alice_jellybeans : color → ℕ)
    (carl_jellybeans : color → ℕ),
  (alice_jellybeans 2 + 2 + 1) = 5 →
  (carl_jellybeans 3 + 2 + 1) = 6 →
  (alice_jellybeans green)/5 * (carl_jellybeans green)/6 +
  (alice_jellybeans red)/5 * (carl_jellybeans red)/6 =
  4 / 15 :=
by
  sorry

end probability_colors_match_l508_508720


namespace kimberly_initial_skittles_l508_508199

theorem kimberly_initial_skittles (total new initial : ℕ) (h1 : total = 12) (h2 : new = 7) (h3 : total = initial + new) : initial = 5 :=
by {
  -- Using the given conditions to form the proof
  sorry
}

end kimberly_initial_skittles_l508_508199


namespace total_cases_of_cat_food_sold_l508_508296

theorem total_cases_of_cat_food_sold :
  (let first_eight := 8 * 3 in
   let next_four := 4 * 2 in
   let last_eight := 8 * 1 in
   first_eight + next_four + last_eight = 40) :=
by
  -- Given conditions:
  -- first_8_customers: 8 customers bought 3 cases each
  -- second_4_customers: 4 customers bought 2 cases each
  -- last_8_customers: 8 customers bought 1 case each
  let first_eight := 8 * 3
  let next_four := 4 * 2
  let last_eight := 8 * 1
  -- Sum of all cases
  show first_eight + next_four + last_eight = 40
  sorry

end total_cases_of_cat_food_sold_l508_508296


namespace problem_statement_l508_508058

-- Condition definitions
def numerator : ℚ := ∑ k in Finset.range 2023, (2024 - (k + 1)) / (k + 1)
def denominator : ℚ := ∑ k in Finset.range (2024 - 2 + 1), 1 / (k + 2)

-- The statement to prove
theorem problem_statement : 
  (numerator / denominator) = 2024 := 
by sorry

end problem_statement_l508_508058


namespace solve_sum_factorials_is_perfect_power_l508_508419

-- Defining what it means for a number to be a perfect power
def is_perfect_power (m : ℕ) : Prop :=
  ∃ (k n : ℕ), k ≥ 2 ∧ m = n^k

-- The condition given is that n is a positive integer
def sum_factorials (n : ℕ) : ℕ :=
  (Finset.range n).sum (λ i, i.fact)

theorem solve_sum_factorials_is_perfect_power (n : ℕ) :
  0 < n → is_perfect_power (sum_factorials n) ↔ n = 1 ∨ n = 3 := by
  -- Proof required here
  sorry

end solve_sum_factorials_is_perfect_power_l508_508419


namespace day_250_of_year_2001_falls_on_tuesday_l508_508165

def days_of_week : Type := {s : String // s = "Sunday" ∨ s = "Monday" ∨ s = "Tuesday" ∨ s = "Wednesday" ∨ s = "Thursday" ∨ s = "Friday" ∨ s = "Saturday"}

noncomputable def day_of_year_2001_day_35 : days_of_week := ⟨"Wednesday", by decide⟩

noncomputable def is_not_leap_year_2001 : Prop := by {
  -- Not divisible by 4
  sorry
}

theorem day_250_of_year_2001_falls_on_tuesday (h1 : day_of_year_2001_day_35 = ⟨"Wednesday", _⟩)
  (h2 : is_not_leap_year_2001) : day_of_week := ⟨"Tuesday", by decide⟩ :=
sorry

end day_250_of_year_2001_falls_on_tuesday_l508_508165


namespace total_cost_price_l508_508692

theorem total_cost_price (P_ct P_ch P_bs : ℝ) (h1 : 8091 = P_ct * 1.24)
    (h2 : 5346 = P_ch * 1.18 * 0.95) (h3 : 11700 = P_bs * 1.30) : 
    P_ct + P_ch + P_bs = 20295 := 
by 
    sorry

end total_cost_price_l508_508692


namespace positive_integer_pairs_l508_508077

theorem positive_integer_pairs (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) :
  (∃ k : ℕ, k > 0 ∧ k = a^2 / (2 * a * b^2 - b^3 + 1)) ↔ 
  (∃ l : ℕ, 0 < l ∧ 
    ((a = 2 * l ∧ b = 1) ∨ (a = l ∧ b = 2 * l) ∨ (a = 8 * l^4 - l ∧ b = 2 * l))) :=
by
  sorry

end positive_integer_pairs_l508_508077


namespace yuan_exchange_l508_508908

theorem yuan_exchange : 
  ∃ (n : ℕ), n = 5 ∧ ∀ (x y : ℕ), x + 5 * y = 20 → x ≥ 0 ∧ y ≥ 0 :=
by {
  sorry
}

end yuan_exchange_l508_508908


namespace triangle_area_l508_508315

theorem triangle_area :
  let A := (2, 2)
  let B := (8, 2)
  let C := (5, 9)
  let base := (B.1 - A.1 : ℕ)
  let height := (C.2 - A.2 : ℕ)
  let area := (1/2 : ℚ) * base * height
  area = 21 :=
by
  let A := (2, 2)
  let B := (8, 2)
  let C := (5, 9)
  let base := (B.1 - A.1 : ℕ)
  let height := (C.2 - A.2 : ℕ)
  let area := (1/2 : ℚ) * base * height
  have h_base : base = 6 := by simp [base]
  have h_height : height = 7 := by simp [height]
  have h_area : area = (1/2 : ℚ) * 6 * 7 := by rw [h_base, h_height]
  have h_final : (1/2 : ℚ) * 6 * 7 = 21 := by norm_num
  rw [h_area, h_final]
  sorry

end triangle_area_l508_508315


namespace radar_placement_and_coverage_area_l508_508650

noncomputable def max_distance (n : ℕ) (r : ℝ) (w : ℝ) : ℝ :=
  (15 : ℝ) / Real.sin (Real.pi / n)

noncomputable def coverage_area (n : ℕ) (r : ℝ) (w : ℝ) : ℝ :=
  (480 : ℝ) * Real.pi / Real.tan (Real.pi / n)

theorem radar_placement_and_coverage_area 
  (n : ℕ) (r w : ℝ) (hn : n = 8) (hr : r = 17) (hw : w = 16) :
  max_distance n r w = (15 : ℝ) / Real.sin (Real.pi / 8) ∧
  coverage_area n r w = (480 : ℝ) * Real.pi / Real.tan (Real.pi / 8) :=
by
  sorry

end radar_placement_and_coverage_area_l508_508650


namespace single_point_domain_of_g5_l508_508979

noncomputable def g1 (x : ℝ) : ℝ := real.sqrt (2 - x)

noncomputable def g (n : ℕ) (x: ℝ) : ℝ :=
  if n = 1 then g1 x
  else (g (n - 1) (real.sqrt ((n + 1) ^ 2 - x)))

theorem single_point_domain_of_g5:
  let M := 5
  let d := -589
  ∃ d : ℝ, (g M d ≠ 0) ∧ (∀ y : ℝ, y ≠ d → g M y = 0) :=
begin
  sorry
end

end single_point_domain_of_g5_l508_508979


namespace conditional_probability_l508_508807

theorem conditional_probability (P : Set (Set ℝ) → ℝ) (A B : Set ℝ) 
  (h1 : P A = 2/5) (h2 : P (A ∩ B) = 1/3) : P (B|A) = 5/6 :=
by
  sorry

end conditional_probability_l508_508807


namespace polar_coordinates_l508_508489

open Real

theorem polar_coordinates (x y : ℝ) (h1 : x = 1 / 2) (h2 : y = - (sqrt 3) / 2) :
    let r := sqrt (x^2 + y^2)
    let θ := atan2 y x
    r = 1 ∧ θ = 5 * π / 3 :=
by
  have hx : x^2 = (1 / 2)^2 := by rw [h1]; norm_num
  have hy : y^2 = ((-(sqrt 3) / 2)^2) := by rw [h2]; norm_num
  calc
    r = sqrt (x^2 + y^2) := by unfold r
    _ = sqrt ((1 / 2)^2 + (-(sqrt 3) / 2)^2) := by rw [hx, hy]
    _ = sqrt (1 / 4 + (3 / 4)) := by norm_num
    _ = sqrt 1 := by norm_num
    _ = 1 := by norm_num

  calc 
    θ = atan2 y x := by unfold θ
    _ = atan2 (-(sqrt 3) / 2) (1 / 2) := by rw [h2, h1]
    _ = -π / 3 := by norm_num
    _ = 5 * π / 3 := by linarith

end polar_coordinates_l508_508489


namespace centroids_coincide_example_l508_508248

variables {A B C A1 B1 C1 : Type}
variables [affine_space ℝ (Type B)] [affine_space ℝ (Type C)] [affine_space ℝ (Type A1)]
variables [affine_space ℝ (Type B1)] [affine_space ℝ (Type C1)]
variables (A B C A1 B1 C1 : points)
variables (k : ℝ)
variables (BC AC AB : lines)

def divides_segment (P Q R : points) (k : ℝ) : Prop :=
  (∃ (t : ℝ), t > 0 ∧ t = k / (1 + k) ∧ R = (1 - t) • P + t • Q)

noncomputable def centroid (P Q R : points) : points :=
  (P + Q + R) / 3

def centroid_coincide (A A1 B B1 C C1 : points) (k : ℝ) : Prop :=
        divides_segment B C A1 k ∧ divides_segment C A B1 k ∧ divides_segment A B C1 k → 
            centroid A B C = centroid A1 B1 C1

axiom centroid_theorem : ∀ (A B C A1 B1 C1 : points) (k : ℝ) (BC AC AB : lines),
  divides_segment B C A1 k ∧ divides_segment C A B1 k ∧ divides_segment A B C1 k →
  centroid A B C = centroid A1 B1 C1

theorem centroids_coincide_example : centroid_coincide A B C A1 B1 C1 k :=
begin
  apply centroid_theorem,
  sorry
end


end centroids_coincide_example_l508_508248


namespace largest_cos_a_of_angles_l508_508566

noncomputable def solve_for_cos_a (a b c : ℝ) : ℝ :=
  sqrt ((3 - sqrt 5) / 2)

theorem largest_cos_a_of_angles (a b c : ℝ) 
  (h1 : Real.sin a = Real.cot b) 
  (h2 : Real.sin b = Real.cot c)
  (h3 : Real.sin c = Real.cot a) : 
  Real.cos a = solve_for_cos_a a b c := 
  sorry

end largest_cos_a_of_angles_l508_508566


namespace percentage_of_area_l508_508007

noncomputable def side_length (s : ℝ) : Prop := s > 0

noncomputable def area_square (s : ℝ) : ℝ := s^2

noncomputable def area_equilateral_triangle (s : ℝ) : ℝ := (sqrt 3 / 4) * s^2

noncomputable def area_pentagon (s : ℝ) : ℝ :=
  area_square s + area_equilateral_triangle s

noncomputable def percentage_area_triangle (s : ℝ) : ℝ :=
  (area_equilateral_triangle s / area_pentagon s) * 100

theorem percentage_of_area (s : ℝ) (h : side_length s) :
  percentage_area_triangle s = (100 * sqrt 3) / (4 + sqrt 3) :=
sorry

end percentage_of_area_l508_508007


namespace solve_equation_l508_508610

theorem solve_equation (x : ℝ) : (x + 2) * (x + 1) = 3 * (x + 1) ↔ (x = -1 ∨ x = 1) :=
by sorry

end solve_equation_l508_508610


namespace min_rods_for_ten_sheep_l508_508358

-- Defining the basic setup, assuming each rod is a unit length
def rods_for_square (n : ℕ) : ℕ := 4 * n

-- Defining the area each rod unit encloses
def area_per_rod (rods : ℕ) : ℕ :=
  if rods = 4 then 1
  else if rods = 12 then 11 -- Based on dodecagon approximation from solution
  else 0 -- Other cases not considered

-- Main theorem statement
theorem min_rods_for_ten_sheep : ∀ n : ℕ, area_per_rod (rods_for_square n) ≥ 10 → n ≥ 3:=
begin
  intro n,
  intro h,
  sorry, -- Proof to be filled in later
end

end min_rods_for_ten_sheep_l508_508358


namespace checkered_rectangles_one_gray_cell_l508_508146

theorem checkered_rectangles_one_gray_cell {m n : ℕ} (gray_cells : ℕ) : m = 2 ∧ n = 20 ∧ gray_cells = 40 →
  (∃ (rectangles : ℕ), rectangles = 176) :=
by
  intros h
  cases h with hm hrest
  cases hrest with hn hgray
  sorry

end checkered_rectangles_one_gray_cell_l508_508146


namespace total_cases_sold_is_correct_l508_508302

-- Define the customer groups and their respective number of cases bought
def n1 : ℕ := 8
def k1 : ℕ := 3
def n2 : ℕ := 4
def k2 : ℕ := 2
def n3 : ℕ := 8
def k3 : ℕ := 1

-- Define the total number of cases sold
def total_cases_sold : ℕ := n1 * k1 + n2 * k2 + n3 * k3

-- The proof statement that the total cases sold is 40
theorem total_cases_sold_is_correct : total_cases_sold = 40 := by
  -- Proof content will be provided here.
  sorry

end total_cases_sold_is_correct_l508_508302


namespace find_a2_an_le_2an_next_sum_bounds_l508_508100

variable {a : ℕ → ℝ}
variable (S : ℕ → ℝ)

-- Given conditions
axiom seq_condition (n : ℕ) (h_pos : a n > 0) : 
  a n ^ 2 + a n = 3 * (a (n + 1)) ^ 2 + 2 * a (n + 1)
axiom a1_condition : a 1 = 1

-- Question 1: Prove the value of a2
theorem find_a2 : a 2 = (Real.sqrt 7 - 1) / 3 :=
  sorry

-- Question 2: Prove a_n ≤ 2 * a_{n+1} for any n ∈ N*
theorem an_le_2an_next (n : ℕ) (h_n : n > 0) : a n ≤ 2 * a (n + 1) :=
  sorry

-- Question 3: Prove 2 - 1 / 2^(n - 1) ≤ S_n < 3 for any n ∈ N*
theorem sum_bounds (n : ℕ) (h_n : n > 0) : 
  2 - 1 / 2 ^ (n - 1) ≤ S n ∧ S n < 3 :=
  sorry

end find_a2_an_le_2an_next_sum_bounds_l508_508100


namespace sum_of_first_twelve_terms_l508_508433

-- Define the arithmetic sequence with given conditions
variable {a d : ℚ}

-- The fifth term of the sequence
def a5 : ℚ := a + 4 * d

-- The seventeenth term of the sequence
def a17 : ℚ := a + 16 * d

-- Sum of the first twelve terms of the arithmetic sequence
def S12 (a d : ℚ) : ℚ := 6 * (2 * a + 11 * d)

theorem sum_of_first_twelve_terms (a : ℚ) (d : ℚ) (h₁ : a5 = 1) (h₂ : a17 = 18) :
  S12 a d = 37.5 := by
  sorry

end sum_of_first_twelve_terms_l508_508433


namespace div_poly_iff_l508_508604

-- Definitions from conditions
def P (x : ℂ) (n : ℕ) := x^(4*n) + x^(3*n) + x^(2*n) + x^n + 1
def Q (x : ℂ) := x^4 + x^3 + x^2 + x + 1

-- The main theorem stating the problem
theorem div_poly_iff (n : ℕ) : 
  ∀ x : ℂ, (P x n) ∣ (Q x) ↔ n % 5 ≠ 0 :=
by sorry

end div_poly_iff_l508_508604


namespace digit_is_4_l508_508264

noncomputable def is_multiple_of_6 (n : ℕ) : Prop :=
  (n % 6 = 0)

noncomputable def is_multiple_of_2 (n : ℕ) : Prop :=
  (n % 2 = 0)

noncomputable def is_multiple_of_3 (n : ℕ) : Prop :=
  (n % 3 = 0)

theorem digit_is_4 (d : ℕ) (h1 : d ∈ {0, 2, 4, 6, 8}) (h2 : is_multiple_of_6 (52280 + d)) : d = 4 :=
by {
  have h3 : is_multiple_of_2 (52280 + d) := by sorry,
  have h4 : is_multiple_of_3 (52280 + d) := by sorry,
  have h5 : (5 + 2 + 2 + 8 + d) % 3 = 0 := by sorry,
  have h6 : (17 + d) % 3 = 0 := h5,
  fin_cases h1,
  case 0 {
    contradiction,
  },
  case 2 {
    contradiction,
  },
  case 4 {
    trivial,
  },
  case 6 {
    contradiction,
  },
  case 8 {
    contradiction,
  }
}

end digit_is_4_l508_508264


namespace max_value_x2_y2_l508_508842

noncomputable def max_x2_y2 (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x^2 + y ≥ x^3 + y^2) : ℝ := 2

theorem max_value_x2_y2 (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^2 + y ≥ x^3 + y^2) : 
  x^2 + y^2 ≤ max_x2_y2 x y hx hy h :=
by
  sorry

end max_value_x2_y2_l508_508842


namespace solve_system_of_equations_l508_508613

theorem solve_system_of_equations (x y : ℝ) 
  (h1 : 6.751 * x + 3.249 * y = 26.751) 
  (h2 : 3.249 * x + 6.751 * y = 23.249) : 
  x = 3 ∧ y = 2 := 
sorry

end solve_system_of_equations_l508_508613


namespace net_transaction_result_l508_508716

theorem net_transaction_result (selling_price_villa selling_price_restaurant : ℝ) (loss_villa gain_restaurant : ℝ) :
  selling_price_villa = 15000 ∧ selling_price_restaurant = 15000 →
  loss_villa = 0.25 ∧ gain_restaurant = 0.15 →
  let cost_villa := selling_price_villa / (1 - loss_villa) in
  let cost_restaurant := selling_price_restaurant / (1 + gain_restaurant) in
  let total_cost := cost_villa + cost_restaurant in
  let total_selling_price := selling_price_villa + selling_price_restaurant in
  total_cost - total_selling_price = 3043.48 :=
by
  sorry

end net_transaction_result_l508_508716


namespace count_equilateral_triangles_in_T_l508_508210

def is_equilateral_triangle (a b c : (ℤ × ℤ × ℤ)) : Prop :=
  let d1 := (a.1 - b.1)^2 + (a.2 - b.2)^2 + (a.3 - b.3)^2
  let d2 := (b.1 - c.1)^2 + (b.2 - c.2)^2 + (b.3 - c.3)^2
  let d3 := (c.1 - a.1)^2 + (c.2 - a.2)^2 + (c.3 - a.3)^2
  d1 = 2 ∧ d1 = d2 ∧ d2 = d3

def T : set (ℤ × ℤ × ℤ) := { p | p.1 ∈ {-1, 0, 1} ∧ p.2 ∈ {-1, 0, 1} ∧ p.3 ∈ {-1, 0, 1} }

theorem count_equilateral_triangles_in_T : 
  {triple | ∃ a b c : (ℤ × ℤ × ℤ), 
    a ∈ T ∧ b ∈ T ∧ c ∈ T ∧ 
    is_equilateral_triangle a b c ∧ 
    triple = (a, b, c)
  }.card = 12 := 
  sorry

end count_equilateral_triangles_in_T_l508_508210


namespace never_return_to_A_l508_508532

open Classical

variables {City : Type} [Fintype City] [DecidableEq City] (d : City → City → ℝ)

-- Distinct pairwise distances
def distinct_pairwise_distances (d : City → City → ℝ) : Prop :=
  ∀ (x y z w : City), (d x y = d z w) → ({x, y} = {z, w} ∨ x = y ∧ z = w)

-- Farthest city function
def farthest (c : City) : City :=
  (argmax (fun x => if x = c then 0 else d c x) {x | x ≠ c}).val

variable (A : City)

-- Initial conditions
axiom h_distinct : distinct_pairwise_distances d
axiom h_start : ∀ c ∈ {x : City | x ≠ A}, d A c < d A (farthest A)
axiom h_next : ∀ c ∈ {x : City | x ≠ farthest A}, d (farthest A) c < d (farthest A) (farthest (farthest A))

-- Lean statement
theorem never_return_to_A (C : City) (h : C ≠ A) : ∀ n : ℕ, (nat.iterate farthest n A) ≠ A :=
by
  sorry

end never_return_to_A_l508_508532


namespace henry_payment_proof_l508_508877

-- Given definitions
def P : ℝ -- amount Henry is paid to paint a bike
def payment_to_sell := P + 8
def total_payment_for_8_bikes := 8 * (P + (P + 8))

-- Prove the goal
theorem henry_payment_proof : total_payment_for_8_bikes = 144 → P = 5 := by
  sorry

end henry_payment_proof_l508_508877


namespace common_tangent_parallel_to_BC_l508_508176

variables {A B C D X : Point}
variables {ABC : Triangle A B C}
variables {circumcircle_ABC : Circle A B C}
variables {circumcircle_BDX : Circle B D X}
variables {circumcircle_CDX : Circle C D X}

-- Given conditions
axiom point_on_side_BC : D ∈ line B C
axiom equality_AB_BD_DC_CA : distance A B + distance B D = distance D C + distance C A
axiom AD_meets_circumcircle_ABC_at_X : (A, D) ∈ circumcircle_ABC ∧ X ≠ A ∧ X ∈ circumcircle_ABC

-- Proof objective
theorem common_tangent_parallel_to_BC :
  exists tangent : Line,
    (∀ circle ∈ {circumcircle_BDX, circumcircle_CDX}, tangent ∈ tangent_line circle) ∧
    tangent ∥ line B C := 
sorry

end common_tangent_parallel_to_BC_l508_508176


namespace calculate_expression_l508_508758

theorem calculate_expression :
  -2^3 * (-3)^2 / (9 / 8) - abs (1 / 2 - 3 / 2) = -65 :=
by
  sorry

end calculate_expression_l508_508758


namespace complement_intersection_l508_508494

def M : Set ℝ := { x | x^2 - 4 > 0 }
def N : Set ℝ := { x | log x / log 2 < 1 }

theorem complement_intersection :
  (Set.univ \ M) ∩ N = { x | 0 < x ∧ x < 2 } := by
  sorry

end complement_intersection_l508_508494


namespace irrational_pi_l508_508326

theorem irrational_pi : 
  (irrational 0.1 = false) ∧
  (irrational (1 / 3) = false) ∧
  (irrational (∛8) = false) ∧
  (irrational π = true) :=
by
  sorry

end irrational_pi_l508_508326


namespace find_number_l508_508171

theorem find_number (n x : ℤ)
  (h1 : (2 * x + 1) = (x - 7)) 
  (h2 : ∃ x : ℤ, n = (2 * x + 1) ^ 2) : 
  n = 25 := 
sorry

end find_number_l508_508171


namespace triangle_inequality_l508_508825

theorem triangle_inequality
  (A B C P D E : Point)
  (h1 : Triangle A B C)
  (h2 : AcuteAngle A B C)
  (h3 : Inside P (Triangle A B C))
  (h4 : ∠APB = 120 ∧ ∠BPC = 120 ∧ ∠CPA = 120)
  (h5 : LineThrough B P ∩ AC = D)
  (h6 : LineThrough C P ∩ AB = E) :
  distance A B + distance A C ≥ 4 * distance D E :=
sorry

end triangle_inequality_l508_508825


namespace round_2_7982_to_0_01_l508_508733

theorem round_2_7982_to_0_01 : round_to (2.7982) (0.01) = 2.80 :=
by
  sorry

end round_2_7982_to_0_01_l508_508733


namespace problem1_problem2_l508_508814

def f (x a : ℝ) : ℝ := abs (1 - x - a) + abs (2 * a - x)

theorem problem1 (a : ℝ) (h : f 1 a < 3) : -2/3 < a ∧ a < 4/3 :=
  sorry

theorem problem2 (a x : ℝ) (h : a ≥ 2/3) : f x a ≥ 1 :=
  sorry

end problem1_problem2_l508_508814


namespace diff_of_squares_l508_508524

theorem diff_of_squares (x y : ℝ) (h1 : x + y = 5) (h2 : x - y = 10) : x^2 - y^2 = 50 := by
  sorry

end diff_of_squares_l508_508524


namespace a_2015_lt_5_l508_508641

noncomputable theory
open Real

def a : ℕ → ℝ
| 0     := 1  -- since a_1 is 1 in the problem statement
| (n+1) := (1 + a n + a n * b n) / b n

def b : ℕ → ℝ
| 0     := 2  -- since b_1 is 2 in the problem statement
| (n+1) := (1 + b n + a n * b n) / a n

theorem a_2015_lt_5 : a 2014 < 5 :=  -- given conditions, we need to prove a2015 < 5
sorry

end a_2015_lt_5_l508_508641


namespace dot_product_eq_sqrt_three_l508_508112

variables (a b : ℝ → ℝ)
variables (θ : ℝ)
variables (norm_a norm_b : ℝ)

-- Given Conditions
axiom angle_condition : θ = π / 6
axiom norm_a_condition : norm_a = 2
axiom norm_b_condition : norm_b = 1

-- Theorem statement to be proven
theorem dot_product_eq_sqrt_three :
  (∥a∥ = norm_a) → (∥b∥ = norm_b) → (∠(a, b) = θ) → (a ⬝ b = √3) :=
by sorry

end dot_product_eq_sqrt_three_l508_508112


namespace problem_1_problem_2_l508_508191

-- Define the given conditions in the problem
def vector_m (a b : ℝ) : ℝ × ℝ := (a, real.sqrt 3 * b)
def vector_n (A B : ℝ) : ℝ × ℝ := (real.cos A, real.sin B)

-- Define the function that checks if two vectors are parallel
def parallel (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.2 = v1.2 * v2.1

-- Problem (1)
def find_A (a b : ℝ) (A B : ℝ) : Prop :=
  parallel (vector_m a b) (vector_n A B) → A = real.pi / 3

-- Problem (2)
def area_of_triangle (a b : ℝ) (A : ℝ) : ℝ :=
  0.5 * b * real.sqrt (a^2 - b^2 / 4) * real.sin A

-- Problem (2) specific case for given values a = sqrt(7), b = 2
def specific_area_of_triangle (a b A : ℝ) : Prop :=
  a = real.sqrt 7 ∧ b = 2 ∧ A = real.pi / 3 →
  area_of_triangle a b A = 3 * real.sqrt 3 / 2

-- Lean statements combining both problems

theorem problem_1 (a b A B : ℝ) :
  parallel (vector_m a b) (vector_n A B) → A = real.pi / 3 := sorry

theorem problem_2 (a b A : ℝ) :
  a = real.sqrt 7 ∧ b = 2 ∧ A = real.pi / 3 →
  area_of_triangle a b A = 3 * real.sqrt 3 / 2 := sorry

end problem_1_problem_2_l508_508191


namespace exists_local_symmetry_point_cubic_exists_local_symmetry_point_exponential_l508_508454

noncomputable theory

open Classical

-- First problem statement
theorem exists_local_symmetry_point_cubic (a b c : ℝ) : ∃ x0 : ℝ, (ax^3 + bx^2 + cx - b) = ax^3 + bx^2 + cx - b → (-ax^3 + bx^2 - cx - b) = - (ax^3 + bx^2 + cx - b) :=
sorry

-- Second problem statement
theorem exists_local_symmetry_point_exponential (m : ℝ) : -2 ≤ m ∧ m ≤ 2 → (∃ x0 : ℝ, 4^x - m * 2^(x + 1) + m^2 - 3 = 4^x - m * 2^(x + 1) + m^2 - 3 ∧ 4^(-x) - m * 2^(-x + 1) + m^2 - 3 = - (4^x - m * 2^(x + 1) + m^2 - 3)) :=
sorry

end exists_local_symmetry_point_cubic_exists_local_symmetry_point_exponential_l508_508454


namespace envelope_weight_l508_508031

theorem envelope_weight :
  (7.225 * 1000) / 850 = 8.5 :=
by
  sorry

end envelope_weight_l508_508031


namespace scientific_notation_correct_l508_508921

theorem scientific_notation_correct (n : ℕ) (h : n = 11580000) : n = 1.158 * 10^7 := 
sorry

end scientific_notation_correct_l508_508921


namespace bob_weekly_profit_l508_508754

-- Definitions for the conditions mentioned
def cost_per_muffin := 0.75
def selling_price_per_muffin := 1.5
def muffins_per_day := 12
def days_per_week := 7

-- Define the theorem to state the weekly profit calculation
theorem bob_weekly_profit : 
  let profit_per_muffin := selling_price_per_muffin - cost_per_muffin
  let total_daily_profit := profit_per_muffin * muffins_per_day
  let weekly_profit := total_daily_profit * days_per_week
  weekly_profit = 63 := 
by 
  sorry


end bob_weekly_profit_l508_508754


namespace change_given_l508_508744

-- Define the given conditions
def oranges_cost := 40
def apples_cost := 50
def mangoes_cost := 60
def initial_amount := 300

-- Calculate total cost of fruits
def total_fruits_cost := oranges_cost + apples_cost + mangoes_cost

-- Define the given change
def given_change := initial_amount - total_fruits_cost

-- Prove that the given change is equal to 150
theorem change_given (h_oranges : oranges_cost = 40)
                     (h_apples : apples_cost = 50)
                     (h_mangoes : mangoes_cost = 60)
                     (h_initial : initial_amount = 300) :
  given_change = 150 :=
by
  -- Proof is omitted, indicated by sorry
  sorry

end change_given_l508_508744


namespace find_y_coordinate_of_P_l508_508579

/-- Let points A, B, C, D and P such that: 
    - A = (-4, -1)
    - B = (-3, 2)
    - C = (3, 2)
    - D = (4, -1)
    - PA + PD = PB + PC = 10
    Assuming the x-coordinate of P is greater than 0, prove that the y-coordinate of P is 2/7 --/
theorem find_y_coordinate_of_P :
  ∃ (P : ℝ × ℝ), (P.1 > 0 ∧
    (real.dist P (-4, -1) + real.dist P (4, -1) = 10) ∧
    (real.dist P (-3, 2) + real.dist P (3, 2) = 10) ∧
    (P.2 = 2 / 7)) :=
sorry

end find_y_coordinate_of_P_l508_508579


namespace triangle_inequality_l508_508225

variables {a b c S : ℝ}
variables {α_1 β_1 γ_1 : ℝ}

-- Suppose a, b, c are the sides of a triangle with area S
-- and α_1, β_1, γ_1 are the angles of another triangle
theorem triangle_inequality (
  h_triangle_sides : a > 0 ∧ b > 0 ∧ c > 0,
  h_area : S > 0,
  h_angles : 0 < α_1 ∧ α_1 < π ∧ 0 < β_1 ∧ β_1 < π ∧ 0 < γ_1 ∧ γ_1 < π ∧ α_1 + β_1 + γ_1 = π
) :
  a^2 * Real.cot α_1 + b^2 * Real.cot β_1 + c^2 * Real.cot γ_1 ≥ 4 * S :=
sorry

end triangle_inequality_l508_508225


namespace sum_sqrt_inequality_l508_508219

theorem sum_sqrt_inequality (n : ℕ) (h : n > 0) :
  (∑ k in Finset.range n, (k + 1) * Real.sqrt (n + 1 - (k + 1))) ≤ (n * (n + 1) * Real.sqrt (2 * n + 1) / (2 * Real.sqrt 3)) :=
sorry

end sum_sqrt_inequality_l508_508219


namespace pentagon_area_l508_508274

noncomputable def angle_F := 100
noncomputable def angle_G := 100
noncomputable def JF := 3
noncomputable def FG := 3
noncomputable def GH := 3
noncomputable def HI := 5
noncomputable def IJ := 5
noncomputable def area_FGHIJ := 9 * Real.sqrt 3 + Real.sqrt 17.1875

theorem pentagon_area : area_FGHIJ = 9 * Real.sqrt 3 + Real.sqrt 17.1875 :=
by
  sorry

end pentagon_area_l508_508274


namespace adolfo_tower_blocks_l508_508028

theorem adolfo_tower_blocks (initial_blocks added_blocks total_blocks : ℝ)
  (h_initial : initial_blocks = 35.0)
  (h_added : added_blocks = 65.0) :
  total_blocks = initial_blocks + added_blocks →
  total_blocks = 100.0 :=
by
  intro h
  rw [h_initial, h_added] at h
  simp at h
  exact h

end adolfo_tower_blocks_l508_508028


namespace part1_part2_l508_508986

noncomputable def f (x : ℝ) : ℝ := log x - (2 * x) / (x + 2)
noncomputable def g (x : ℝ) : ℝ := f x - 4 / (x + 2)

theorem part1 (x : ℝ) (hx : x > 0) : 
  (has_deriv_at f (x^2 + 4) / (x * (x + 2)^2) x) ∧ ((x^2 + 4) / (x * (x + 2)^2) > 0) :=
sorry

theorem part2 (a : ℝ) (hx : ∀ x : ℝ, x > 0 → g x < x + a) : a > -3 :=
sorry

end part1_part2_l508_508986


namespace exists_M_equal_angles_l508_508103

noncomputable def find_M (A B C D : ℝ × ℝ) : Prop :=
  ∃ (M : ℝ × ℝ),
    angle A M B = angle B M C ∧
    angle B M C = angle C M D

-- Let A, B, C, and D be four points on a straight line.
variable (A B C D : ℝ × ℝ)

-- The Lean statement:
theorem exists_M_equal_angles (collinear : collinear {A, B, C, D}) :
  find_M A B C D :=
sorry

end exists_M_equal_angles_l508_508103


namespace min_value_quadratic_l508_508393

theorem min_value_quadratic : 
  ∀ x : ℝ, (4 * x^2 - 12 * x + 9) ≥ 0 :=
by
  sorry

end min_value_quadratic_l508_508393


namespace scientific_notation_11580000_l508_508913

theorem scientific_notation_11580000 :
  (11580000 : ℝ) = 1.158 * 10^7 :=
sorry

end scientific_notation_11580000_l508_508913


namespace A_and_B_to_complete_work_l508_508699

def A_rate (B_rate : ℝ) : ℝ := 3 * B_rate
def A_rate_alone : ℝ := 1 / 20
def B_rate (A_rate_alone : ℝ) : ℝ := A_rate_alone / 3
def Combined_rate (A_rate B_rate : ℝ) : ℝ := A_rate + B_rate

theorem A_and_B_to_complete_work (B_rate : ℝ) (h1 : A_rate B_rate = 1 / 20) :
  1 / Combined_rate (A_rate B_rate) B_rate = 15 :=
by
  sorry

end A_and_B_to_complete_work_l508_508699


namespace cube_minimum_yellow_surface_area_l508_508691

open nat

/-- 
Given a cube with 4-inch edges constructed from 64 smaller 1-inch cubes, 
where 48 cubes are blue and 16 cubes are yellow. Prove that the smallest 
possible fraction of the surface area that is yellow is 1/12.
-/
theorem cube_minimum_yellow_surface_area 
  (total_cubes : ℕ := 64) 
  (blue_cubes : ℕ := 48) 
  (yellow_cubes : ℕ := 16)
  (edge_length : ℕ := 4) 
  (small_cube_edge : ℕ := 1)
  (surface_area : ℕ := 6 * edge_length * edge_length)
  (exposed_yellow_surface : ℕ := 8) :
  (exposed_yellow_surface : ℚ) / (surface_area : ℚ) = 1 / 12 :=
begin
  sorry,
end

end cube_minimum_yellow_surface_area_l508_508691


namespace original_bill_l508_508682

theorem original_bill (n : ℕ) (d : ℝ) (p : ℝ) (B : ℝ) (h1 : n = 5) (h2 : d = 0.06) (h3 : p = 18.8)
  (h4 : 0.94 * B = n * p) :
  B = 100 :=
sorry

end original_bill_l508_508682


namespace even_total_score_probability_l508_508347

theorem even_total_score_probability :
  let rings := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  let scores := [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
  let areas := rings.map (λ n, π * (n^2 - (n - 1)^2))
  let total_area := π * 100
  let even_scores := [10, 8, 6, 4, 2]
  let odd_scores := [9, 7, 5, 3, 1]
  let even_area := (even_scores.map (λ score, areas[(scores.indexOf score)])).sum
  let odd_area := (odd_scores.map (λ score, areas[(scores.indexOf score)])).sum
  let P_even := even_area / total_area
  let P_odd := odd_area / total_area
  let P_even_total := P_even * P_even + P_odd * P_odd
  P_even_total = 5 / 8 :=
sorry

end even_total_score_probability_l508_508347


namespace mine_placement_in_grid_l508_508342

-- Definitions based on the problem's conditions
def isAdjacent (p1 p2 : (ℕ × ℕ)) : Prop :=
  abs (p1.1 - p2.1) ≤ 1 ∧ abs (p1.2 - p2.2) ≤ 1

def grid_has_numbers_with_adjacent_mines (grid : ℕ × ℕ → ℕ) (mines : (ℕ × ℕ) → Prop) : Prop :=
  ∀ (x y : ℕ × ℕ), grid x y ≥ 0 → grid x y = (finset.filter (isAdjacent (x, y)) (finset.univ (ℕ × ℕ))).card (mines x y)

-- Statement of the problem
theorem mine_placement_in_grid :
  ∃ mines : (ℕ × ℕ) → Prop, (grid_has_numbers_with_adjacent_mines grid mines) ∧ (finset.univ (ℕ × ℕ) mines).card = 14 :=
sorry

end mine_placement_in_grid_l508_508342


namespace sum_of_squares_of_distances_l508_508709

variables {A : Type*} [metric_space A]

/-- A statement of the proof problem in Lean 4 -/
theorem sum_of_squares_of_distances
  (n : ℕ) (R : ℝ) (O X : A)
  (vertices : fin n → A)
  (h_regular : regular_polygon vertices R O)
  (h_d : dist O X = d) :
  ∑ i : fin n, (dist (vertices i) X)^2 = n * (R^2 + d^2) :=
sorry

/-- Definition of a regular polygon inscribed in a circle -/
def regular_polygon {A : Type*} [metric_space A] (vertices : fin n → A) (R : ℝ) (O : A) : Prop :=
  ∀ i, dist O (vertices i) = R ∧
  ∀ i j, i ≠ j → dist (vertices i) (vertices j) = 2 * R * sin (π / n) 

end sum_of_squares_of_distances_l508_508709


namespace cliff_collection_has_180_rocks_l508_508927

noncomputable def cliffTotalRocks : ℕ :=
  let shiny_igneous_rocks := 40
  let total_igneous_rocks := shiny_igneous_rocks * 3 / 2
  let total_sedimentary_rocks := total_igneous_rocks * 2
  total_igneous_rocks + total_sedimentary_rocks

theorem cliff_collection_has_180_rocks :
  let shiny_igneous_rocks := 40
  let total_igneous_rocks := shiny_igneous_rocks * 3 / 2
  let total_sedimentary_rocks := total_igneous_rocks * 2
  total_igneous_rocks + total_sedimentary_rocks = 180 := sorry

end cliff_collection_has_180_rocks_l508_508927


namespace proof_goal_l508_508978

variable (g : ℕ → ℕ)
variable (g_inv : ℕ → ℕ)

axiom g_4_eq_6 : g 4 = 6
axiom g_6_eq_3 : g 6 = 3
axiom g_7_eq_4 : g 7 = 4
axiom g_inv_4_eq_7 : g_inv 4 = 7
axiom g_inv_6_eq_4 : g_inv 6 = 4
axiom g_inv_11_eq_2 : g_inv 11 = 2

theorem proof_goal : g_inv (g_inv 4 + g_inv 6) = 2 :=
by
  rw [g_inv_4_eq_7, g_inv_6_eq_4]
  exact g_inv_11_eq_2

end proof_goal_l508_508978


namespace probability_even_sum_l508_508093

def is_even_sum (a b : ℕ) : Prop := (a + b) % 2 = 0

theorem probability_even_sum :
  let digits := {1, 2, 3, 4, 5}
  let pairs := {(a, b) | a ∈ digits, b ∈ digits, a < b}
  let even_pairs := {(a, b) | (a, b) ∈ pairs, is_even_sum a b}
  (even_pairs.card : ℚ) / (pairs.card : ℚ) = 2 / 5 := by
  sorry

end probability_even_sum_l508_508093


namespace proof_tan_x_l508_508811

theorem proof_tan_x
  (x : ℝ)
  (h1 : 0 < x)
  (h2 : x < π / 2)
  (h3 : sin x - cos x = π / 4) :
  let a := 32
  let b := 16
  let c := 2
  in tan x + 1 / tan x = 32 / (16 - π^2) ∧ a + b + c = 50 :=
by
  sorry

end proof_tan_x_l508_508811


namespace number_of_valid_polynomials_l508_508425

noncomputable def count_polynomials_meeting_conditions : ℕ := sorry

theorem number_of_valid_polynomials :
  count_polynomials_meeting_conditions = 7200 :=
sorry

end number_of_valid_polynomials_l508_508425


namespace graph_passes_through_point_l508_508278

variable (a : ℝ) (x y : ℝ)
hypothesis ha1 : a > 0
hypothesis ha2 : a ≠ 1

theorem graph_passes_through_point :
  (f : ℝ → ℝ) = (fun x => log a (x + 2) - 1) →
  f (-1) = -1 :=
by
  intro hf
  exact sorry

end graph_passes_through_point_l508_508278


namespace sum_of_first_twelve_terms_arithmetic_sequence_l508_508436

-- Definitions
def a (a1 d : ℝ) (n : ℕ) : ℝ := a1 + (n - 1) * d

def Sn (a1 d : ℝ) (n : ℕ) : ℝ := (n / 2) * (2 * a1 + (n - 1) * d)

-- Main Statement
theorem sum_of_first_twelve_terms_arithmetic_sequence  (a1 d : ℝ) 
  (h1 : a a1 d 5 = 1) (h2 : a a1 d 17 = 18) :
  Sn a1 d 12 = 37.5 :=
sorry

end sum_of_first_twelve_terms_arithmetic_sequence_l508_508436


namespace number_of_nephews_number_of_nephews_toy_problem_l508_508655

noncomputable def price_of_candy : ℝ := by sorry
noncomputable def price_of_orange : ℝ := 2 * price_of_candy
noncomputable def price_of_cake : ℝ := 4 * price_of_candy
noncomputable def price_of_chocolate : ℝ := 7 * price_of_candy
noncomputable def price_of_book : ℝ := 14 * price_of_candy

def total_price_of_gift : ℝ :=
  price_of_candy + price_of_orange + price_of_cake + price_of_chocolate + price_of_book

theorem number_of_nephews (total_money_spent : ℝ) (price_of_candy : ℝ) : ℝ :=
let total_money_spent := 224 * price_of_candy in
total_money_spent / total_price_of_gift

theorem number_of_nephews_toy_problem : number_of_nephews 224 price_of_candy = 8 :=
by sorry

end number_of_nephews_number_of_nephews_toy_problem_l508_508655


namespace locus_of_right_angle_vertices_l508_508249

variables {A : Point} {B C : Point}
def segment := line B C

theorem locus_of_right_angle_vertices {A : Point} (BC : segment) :
  locus_of_points (λ X, ∃ P ∈ BC, X ∈ (circle (A, P))) =
    region_enclosed (⋃ P ∈ segment B C, circle (A, P)) :=
by
  sorry

end locus_of_right_angle_vertices_l508_508249


namespace solve_expression_l508_508445

theorem solve_expression (x y : ℝ) (h : (x + y - 2020) * (2023 - x - y) = 2) :
  (x + y - 2020)^2 * (2023 - x - y)^2 = 4 := by
  sorry

end solve_expression_l508_508445


namespace coloring_count_l508_508880

open Nat

def properDivisors (n : ℕ) : List ℕ := (List.range n).filter (λ d => d > 1 ∧ d ∣ n)

noncomputable def colorings : List ℕ → ℕ
| []            => 1
| (n :: ns)     => if n.prime
                   then 4 * colorings ns  -- 4 color choices for primes
                   else match properDivisors n with
                        | []            => 4 * colorings ns -- no proper divisors
                        | divisors      => 
                            let divisorColors := divisors.map (λ d => (List.map (λ x => x ^ (n + d)) [1, 2, 3]).foldr (*) 1)
                            divisorColors.foldr (*) 3 * colorings ns 

theorem coloring_count : colorings [2, 3, 4, 5, 6, 7, 8, 9, 10, 11] = 248832 := by
  sorry

end coloring_count_l508_508880


namespace tan_theta_solution_l508_508398

theorem tan_theta_solution (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < 15) 
  (h_tan_eq : Real.tan θ + Real.tan (2 * θ) + Real.tan (4 * θ) = 0) :
  Real.tan θ = 1 / Real.sqrt 2 :=
sorry

end tan_theta_solution_l508_508398


namespace count_positive_numbers_l508_508535

def is_positive (N : ℕ) : Prop :=
  -- Define the predicate that checks if a number N can be transformed from all minuses
  -- to all pluses as described in the problem statement.
  sorry

theorem count_positive_numbers :
  #{ n | 3 ≤ n ∧ n ≤ 1400 ∧ is_positive n} = 1396 :=
sorry

end count_positive_numbers_l508_508535


namespace pentagon_area_l508_508273

noncomputable def angle_F := 100
noncomputable def angle_G := 100
noncomputable def JF := 3
noncomputable def FG := 3
noncomputable def GH := 3
noncomputable def HI := 5
noncomputable def IJ := 5
noncomputable def area_FGHIJ := 9 * Real.sqrt 3 + Real.sqrt 17.1875

theorem pentagon_area : area_FGHIJ = 9 * Real.sqrt 3 + Real.sqrt 17.1875 :=
by
  sorry

end pentagon_area_l508_508273


namespace find_number_l508_508367

theorem find_number :
  let s := 2615 + 3895
  let d := 3895 - 2615
  let q := 3 * d
  let x := s * q + 65
  x = 24998465 :=
by
  let s := 2615 + 3895
  let d := 3895 - 2615
  let q := 3 * d
  let x := s * q + 65
  sorry

end find_number_l508_508367


namespace circumscribed_sphere_surface_area_l508_508544

theorem circumscribed_sphere_surface_area (PA BC AC BP CP AB : ℝ)
  (h1 : PA = BC ∧ PA = 2 * Real.sqrt 13)
  (h2 : AC = BP ∧ AC = Real.sqrt 41)
  (h3 : CP = AB ∧ CP = Real.sqrt 61) :
  surface_area (circumscribed_sphere P ABC) = 77 * Real.pi :=
by {
  sorry
}

end circumscribed_sphere_surface_area_l508_508544


namespace area_of_triangle_tangent_to_curve_at_point_1_1_l508_508267

-- Definitions and conditions based on the problem statement
def curve (x : ℝ) : ℝ := 3 * x * Real.log x + x

def deriv_curve (x : ℝ) : ℝ := 4 + 3 * Real.log x

def tangent_line (x : ℝ) : ℝ := 4 * x - 3

-- The main theorem stating the problem
theorem area_of_triangle_tangent_to_curve_at_point_1_1 :
  let x_intercept := 3 / 2 in
  let y_intercept := -3 in
  let area := 1 / 2 * x_intercept * (-y_intercept) in
  area = 9 / 4 :=
by
  -- Intermediate steps and the calculation proof should be here
  sorry

end area_of_triangle_tangent_to_curve_at_point_1_1_l508_508267


namespace change_given_l508_508743

-- Define the given conditions
def oranges_cost := 40
def apples_cost := 50
def mangoes_cost := 60
def initial_amount := 300

-- Calculate total cost of fruits
def total_fruits_cost := oranges_cost + apples_cost + mangoes_cost

-- Define the given change
def given_change := initial_amount - total_fruits_cost

-- Prove that the given change is equal to 150
theorem change_given (h_oranges : oranges_cost = 40)
                     (h_apples : apples_cost = 50)
                     (h_mangoes : mangoes_cost = 60)
                     (h_initial : initial_amount = 300) :
  given_change = 150 :=
by
  -- Proof is omitted, indicated by sorry
  sorry

end change_given_l508_508743


namespace sum_of_fractions_l508_508784

theorem sum_of_fractions :
  (∑ n in Finset.range 9, (1 : ℝ) / (n + 2) / (n + 3)) = 9 / 22 := by
  sorry

end sum_of_fractions_l508_508784


namespace probability_four_of_six_same_l508_508608

noncomputable def six_dice_probability : ℚ :=
  let total_outcomes := 6^4 in
  let successful_outcomes := 150 + 20 + 1 + 5 in
  successful_outcomes / total_outcomes

theorem probability_four_of_six_same :
  (six_dice_probability = 11 / 81) :=
by
  -- Proof will be here
  sorry

end probability_four_of_six_same_l508_508608


namespace sum_of_first_twelve_terms_arithmetic_sequence_l508_508435

-- Definitions
def a (a1 d : ℝ) (n : ℕ) : ℝ := a1 + (n - 1) * d

def Sn (a1 d : ℝ) (n : ℕ) : ℝ := (n / 2) * (2 * a1 + (n - 1) * d)

-- Main Statement
theorem sum_of_first_twelve_terms_arithmetic_sequence  (a1 d : ℝ) 
  (h1 : a a1 d 5 = 1) (h2 : a a1 d 17 = 18) :
  Sn a1 d 12 = 37.5 :=
sorry

end sum_of_first_twelve_terms_arithmetic_sequence_l508_508435


namespace f_in_interval_l508_508571

noncomputable def f (x : ℝ) : ℝ := 
  if -2 ≤ x ∧ x ≤ 2 then -x^2 + 1 else sorry -- Placeholder for values outside [-2, 2]

theorem f_in_interval (x : ℝ) (h : -6 ≤ x ∧ x ≤ -2) : 
  f(x) = -(x + 4) ^ 2 + 1 := 
begin
  have h1 : (-2 ≤ x + 4) ∧ (x + 4 ≤ 2) := sorry,
  exact sorry
end

end f_in_interval_l508_508571


namespace inequality_l508_508607

theorem inequality (k n : ℕ) (x : Fin k → ℝ) 
  (h_pos : ∀ i, 0 < x i) 
  (h_sum : (∑ i : Fin k, x i) = 1) : 
  (∑ i : Fin k, (1 / x i ^ n)) ≥ k ^ (n + 1) := 
sorry

end inequality_l508_508607


namespace integer_solutions_eq_l508_508787

theorem integer_solutions_eq (x y : ℤ) (h : y^2 = x^3 + (x + 1)^2) : (x, y) = (0, 1) ∨ (x, y) = (0, -1) :=
by
  sorry

end integer_solutions_eq_l508_508787


namespace layoffs_total_l508_508353

theorem layoffs_total : 
  ∀ (initial_employees : ℕ) (layoff_percentage : ℕ) (rounds : ℕ),
  initial_employees = 1000 →
  layoff_percentage = 10 →
  rounds = 3 →
  let first_round_layoff := initial_employees * layoff_percentage / 100;
      employees_after_first_round := initial_employees - first_round_layoff;
      second_round_layoff := employees_after_first_round * layoff_percentage / 100;
      employees_after_second_round := employees_after_first_round - second_round_layoff;
      third_round_layoff := employees_after_second_round * layoff_percentage / 100;
      total_layoff := first_round_layoff + second_round_layoff + third_round_layoff
  in total_layoff = 271 :=
by
  intros initial_employees layoff_percentage rounds
  intros h1 h2 h3
  let first_round_layoff := initial_employees * layoff_percentage / 100
  let employees_after_first_round := initial_employees - first_round_layoff
  let second_round_layoff := employees_after_first_round * layoff_percentage / 100
  let employees_after_second_round := employees_after_first_round - second_round_layoff
  let third_round_layoff := employees_after_second_round * layoff_percentage / 100
  let total_layoff := first_round_layoff + second_round_layoff + third_round_layoff
  have : total_layoff = 271 := sorry
  exact this

end layoffs_total_l508_508353


namespace prob_no_english_and_history_l508_508198

variables {Ω : Type} [MeasureSpace Ω]
variables (E H : Event Ω) {P : ProbabilityMeasure Ω}

-- Given conditions
def english_test : Prop := P E = 5 / 9
def history_test : Prop := P H = 1 / 3
def independent_events : Prop := Independent E H

-- Question and answer
theorem prob_no_english_and_history :
  english_test E P →
  history_test H P →
  independent_events E H P →
  P (Eᶜ ∩ H) = 4 / 27 :=
by
  intros h1 h2 h3
  -- The proof goes here, but it's omitted for now
  sorry

end prob_no_english_and_history_l508_508198


namespace problem_statement_l508_508057

-- Condition definitions
def numerator : ℚ := ∑ k in Finset.range 2023, (2024 - (k + 1)) / (k + 1)
def denominator : ℚ := ∑ k in Finset.range (2024 - 2 + 1), 1 / (k + 2)

-- The statement to prove
theorem problem_statement : 
  (numerator / denominator) = 2024 := 
by sorry

end problem_statement_l508_508057


namespace a_2n_is_perfect_square_l508_508215

noncomputable def a : ℕ → ℕ
| 0     := 0
| 1     := 1
| 2     := 1
| 3     := 2
| 4     := 4
| (n+5) := a (n+4) + a (n+2) + a n

noncomputable def f : ℕ → ℕ
| 0     := 0
| 1     := 1
| n + 2 := f (n + 1) + f n

theorem a_2n_is_perfect_square (n : ℕ) :
  ∃ f_n : ℕ, a (2 * n) = f_n * f_n :=
sorry

end a_2n_is_perfect_square_l508_508215


namespace proof_statement_l508_508836

-- Assume 5 * 3^x = 243
def condition (x : ℝ) : Prop := 5 * (3:ℝ)^x = 243

-- Define the log base 3 for use in the statement
noncomputable def log_base_3 (y : ℝ) : ℝ := Real.log y / Real.log 3

-- State that if the condition holds, then (x + 2)(x - 2) = 21 - 10 * log_base_3 5 + (log_base_3 5)^2
theorem proof_statement (x : ℝ) (h : condition x) : (x + 2) * (x - 2) = 21 - 10 * log_base_3 5 + (log_base_3 5)^2 := sorry

end proof_statement_l508_508836


namespace intersection_of_A_and_B_l508_508867

open Set

def A : Set ℝ := {x | 0 < x ∧ x < 3}
def B : Set ℝ := {x | x^2 ≥ 4}

theorem intersection_of_A_and_B : A ∩ B = {x | 2 ≤ x ∧ x < 3} :=
  by
    sorry

end intersection_of_A_and_B_l508_508867


namespace gas_station_total_boxes_l508_508697

theorem gas_station_total_boxes :
  ∀ (chocolate_candy sugar_candy gum : ℕ), chocolate_candy = 2 → sugar_candy = 5 → gum = 2 →
  chocolate_candy + sugar_candy + gum = 9 :=
by
  intros chocolate_candy sugar_candy gum h₁ h₂ h₃
  rw [h₁, h₂, h₃]
  norm_num

end gas_station_total_boxes_l508_508697


namespace sin_of_right_angle_l508_508539

theorem sin_of_right_angle (A B C : Type)
  (angle_A : Real) (AB BC : Real)
  (h_angleA : angle_A = 90)
  (h_AB : AB = 16)
  (h_BC : BC = 24) :
  Real.sin (angle_A) = 1 :=
by
  sorry

end sin_of_right_angle_l508_508539


namespace each_cut_piece_weight_l508_508958

theorem each_cut_piece_weight (L : ℕ) (W : ℕ) (c : ℕ) 
  (hL : L = 20) (hW : W = 150) (hc : c = 2) : (L / c) * W = 1500 := by
  sorry

end each_cut_piece_weight_l508_508958


namespace money_sufficient_for_B_wages_l508_508376

-- Let B and C be daily wages
variables (B C : ℝ)

-- Conditions given in the problem
variables (S : ℝ) (hC : S = 24 * C) (hBC : S = 8 * (B + C)) (hB : S = ∀ D, D * B)

-- The theorem to prove
theorem money_sufficient_for_B_wages : ∃ D, D = 12 :=
by {
  -- Define equations for the money conditions
  have h1 : S = 24 * C := hC,
  have h2 : S = 8 * (B + C) := hBC,
  -- Solve for D given B and C wages conditions
  sorry
}

end money_sufficient_for_B_wages_l508_508376


namespace distinct_products_count_220_l508_508580

-- Define the prime factorization of 72000 and T as its divisors
def prime_factors : ℕ → ℕ → ℕ → Prop :=
λ a b c, (2^a * 3^b * 5^c) ∣ 72000 ∧ a ≤ 6 ∧ b ≤ 2 ∧ c ≤ 3

-- Define the total number of distinct products of two elements in T
noncomputable def distinct_products_count (n : ℕ) : Prop :=
  (∀ (x y : ℕ), x ≠ y → (x ∈ T ∧ y ∈ T) → (∃ z, z = x * y ∧ z ∈ { d | d ∣ 72000 })) ∧ n = 220

theorem distinct_products_count_220 : distinct_products_count 220 :=
sorry

end distinct_products_count_220_l508_508580


namespace range_x_sub_sqrt3_y_l508_508830

theorem range_x_sub_sqrt3_y
  (x y : ℝ)
  (h : x^2 + y^2 - 2*x + 2*sqrt(3)*y + 3 = 0) :
  2 ≤ x - sqrt(3) * y ∧ x - sqrt(3) * y ≤ 6 :=
sorry

end range_x_sub_sqrt3_y_l508_508830


namespace zongzi_unit_prices_max_purchase_A_zongzi_l508_508065

theorem zongzi_unit_prices (x : ℝ) (price_A : ℝ) (price_B : ℝ) (quantity_A : ℝ) (quantity_B : ℝ) : 
  price_A = 10 ∧ price_B = 5 ∧ quantity_A = quantity_B - 50 ∧ price_A * quantity_A = 1500 ∧ price_B * quantity_B = 1000 :=
by
  have h1 : price_A = 2 * price_B, 
  have h2 : quantity_A = quantity_B - 50, 
  have h3 : price_A * quantity_A = 1500,
  have h4 : price_B * quantity_B = 1000,
  have h5 : quantity_B = 200,
  have h6 : 10 * quantity_A + 5 * (200 - quantity_A) ≤ 1450,
  sorry
  
theorem max_purchase_A_zongzi (quantity_A : ℕ) : 10 * quantity_A + 5 * (200 - quantity_A) ≤ 1450 → quantity_A ≤ 90 :=
by
  intro h,
  sorry

end zongzi_unit_prices_max_purchase_A_zongzi_l508_508065


namespace remainder_when_a_plus_b_div_40_is_28_l508_508591

theorem remainder_when_a_plus_b_div_40_is_28 :
  ∃ k j : ℤ, (a = 80 * k + 74 ∧ b = 120 * j + 114) → (a + b) % 40 = 28 := by
  sorry

end remainder_when_a_plus_b_div_40_is_28_l508_508591


namespace total_walnut_trees_l508_508291

-- Define the conditions
def current_walnut_trees := 4
def new_walnut_trees := 6

-- State the lean proof problem
theorem total_walnut_trees : current_walnut_trees + new_walnut_trees = 10 := by
  sorry

end total_walnut_trees_l508_508291


namespace equation_of_circle_l508_508451

theorem equation_of_circle
  (tangent_at_y_axis : ∀ P : ℝ × ℝ, P = (0, 3) → P ∈ ({p | (p.1 - a)^2 + (p.2 - 3)^2 = (|a|) ^ 2} : set (ℝ × ℝ)))
  (intersect_segment_length : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (|x1 - x2| = 8) ∧ (0, 0) ∈ ({p | (p.1 - a)^2 + (p.2 - 3)^2 = (|a|) ^ 2} : set (ℝ × ℝ))) :
  (∃ a : ℝ, a = 5 ∨ a = -5) ∧ (∀ (x y : ℝ), (x + 5)^2 + (y - 3)^2 = 25 ∨ (x - 5)^2 + (y - 3)^2 = 25) :=
by sorry

end equation_of_circle_l508_508451


namespace paul_spent_374_43_l508_508600

noncomputable def paul_total_cost_after_discounts : ℝ :=
  let dress_shirts := 4 * 15.00
  let discount_dress_shirts := dress_shirts * 0.20
  let cost_dress_shirts := dress_shirts - discount_dress_shirts
  
  let pants := 2 * 40.00
  let discount_pants := pants * 0.30
  let cost_pants := pants - discount_pants
  
  let suit := 150.00
  
  let sweaters := 2 * 30.00
  
  let ties := 3 * 20.00
  let discount_tie := 20.00 * 0.50
  let cost_ties := 20.00 + (20.00 - discount_tie) + 20.00

  let shoes := 80.00
  let discount_shoes := shoes * 0.25
  let cost_shoes := shoes - discount_shoes

  let total_after_discounts := cost_dress_shirts + cost_pants + suit + sweaters + cost_ties + cost_shoes
  
  let total_after_coupon := total_after_discounts * 0.90
  
  let total_after_rewards := total_after_coupon - (500 * 0.05)
  
  let total_after_tax := total_after_rewards * 1.05
  
  total_after_tax

theorem paul_spent_374_43 :
  paul_total_cost_after_discounts = 374.43 :=
by
  sorry

end paul_spent_374_43_l508_508600


namespace isosceles_triangle_slopes_l508_508497

theorem isosceles_triangle_slopes (k : ℝ) (h₁ : k > 0)
  (h₂ : ∀ l_1 l_2 : ℝ × ℝ → ℝ × ℝ → Prop, 
    ∃ P, 
    (∀ p1 p2, l_1 (p1, p2) (1 - p1 / k, p2)) ∧
    (∀ p1 p2, l_2 (p1, p2) (1 - p1 * (2 * k), p2)) ∧
    (∃ p1 p2, is_isosceles P (p1, p2) (1, 0))) : 
  k = real.sqrt 2 / 2 ∨ k = real.sqrt 2 :=
sorry

end isosceles_triangle_slopes_l508_508497


namespace bob_weekly_profit_l508_508753

theorem bob_weekly_profit :
  let daily_muffins := 12
  let cost_per_muffin := 0.75
  let sell_price_per_muffin := 1.5
  let days_per_week := 7
  let daily_cost := daily_muffins * cost_per_muffin
  let daily_revenue := daily_muffins * sell_price_per_muffin
  let daily_profit := daily_revenue - daily_cost
  let weekly_profit := daily_profit * days_per_week
  in weekly_profit = 63 :=
by
  sorry

end bob_weekly_profit_l508_508753


namespace sqrt11_plus_sqrt3_lt_sqrt9_plus_sqrt5_l508_508045

noncomputable def compare_sq_roots_sum : Prop := 
  (Real.sqrt 11 + Real.sqrt 3) < (Real.sqrt 9 + Real.sqrt 5)

theorem sqrt11_plus_sqrt3_lt_sqrt9_plus_sqrt5 :
  compare_sq_roots_sum :=
sorry

end sqrt11_plus_sqrt3_lt_sqrt9_plus_sqrt5_l508_508045


namespace sum_of_first_twelve_terms_l508_508434

-- Define the arithmetic sequence with given conditions
variable {a d : ℚ}

-- The fifth term of the sequence
def a5 : ℚ := a + 4 * d

-- The seventeenth term of the sequence
def a17 : ℚ := a + 16 * d

-- Sum of the first twelve terms of the arithmetic sequence
def S12 (a d : ℚ) : ℚ := 6 * (2 * a + 11 * d)

theorem sum_of_first_twelve_terms (a : ℚ) (d : ℚ) (h₁ : a5 = 1) (h₂ : a17 = 18) :
  S12 a d = 37.5 := by
  sorry

end sum_of_first_twelve_terms_l508_508434


namespace min_value_on_neg_infinite_l508_508513

def odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f x

def max_value_on_interval (F : ℝ → ℝ) (a b : ℝ) (max_val : ℝ) : Prop :=
∀ x, (0 < x → F x ≤ max_val) ∧ (∃ y, 0 < y ∧ F y = max_val)

theorem min_value_on_neg_infinite (f g : ℝ → ℝ) (a b : ℝ) (F : ℝ → ℝ)
  (h_odd_f : odd_function f) (h_odd_g : odd_function g)
  (h_def_F : ∀ x, F x = a * f x + b * g x + 2)
  (h_max_F_on_0_inf : max_value_on_interval F a b 8) :
  ∃ x, x < 0 ∧ F x = -4 :=
sorry

end min_value_on_neg_infinite_l508_508513


namespace scientific_notation_correct_l508_508919

theorem scientific_notation_correct (n : ℕ) (h : n = 11580000) : n = 1.158 * 10^7 := 
sorry

end scientific_notation_correct_l508_508919


namespace probability_pair_sum_l508_508355

-- Define the given conditions
def initial_deck : List ℕ := List.replicate 4 1 ++ List.replicate 4 2 ++ List.replicate 4 3 ++
                             List.replicate 4 4 ++ List.replicate 4 5 ++ List.replicate 4 6 ++
                             List.replicate 4 7 ++ List.replicate 4 8 ++ List.replicate 4 9 ++
                             List.replicate 4 10 ++ List.replicate 4 11 ++ List.replicate 4 12 ++
                             List.replicate 4 13

def remaining_deck : List ℕ := (initial_deck.drop 2).filter (≠ (initial_deck.nth 0).getD 0)

-- We need a theorem for the problem statement
theorem probability_pair_sum (deck : List ℕ) (len : deck.length = 50) : 
  let total_ways := Nat.choose 50 2
  let pair_ways := 12 * Nat.choose 4 2 + 1
  let probability := pair_ways / total_ways
  let gcd_value := Nat.gcd pair_ways total_ways
  let simplified_fraction_num := pair_ways / gcd_value
  let simplified_fraction_den := total_ways / gcd_value
  let sum_fraction := simplified_fraction_num + simplified_fraction_den
  sum_fraction = 1298 := by
{
    -- Lean proof content would go here
    sorry
}

end probability_pair_sum_l508_508355


namespace infinite_primes_congruent_2_mod_3_l508_508251

theorem infinite_primes_congruent_2_mod_3 :
  ∃^∞ p : ℕ, Prime p ∧ p % 3 = 2 :=
sorry

end infinite_primes_congruent_2_mod_3_l508_508251


namespace min_disks_to_store_files_l508_508951

open Nat

theorem min_disks_to_store_files :
  ∃ minimum_disks : ℕ,
    (minimum_disks = 24) ∧
    ∀ (files : ℕ) (disk_capacity : ℕ) (file_sizes : List ℕ),
      files = 36 →
      disk_capacity = 144 →
      (∃ (size_85 : ℕ) (size_75 : ℕ) (size_45 : ℕ),
         size_85 = 5 ∧
         size_75 = 15 ∧
         size_45 = 16 ∧
         (∀ (disks : ℕ), disks >= minimum_disks →
            ∃ (used_disks_85 : ℕ) (remaining_files_45 : ℕ) (used_disks_45 : ℕ) (used_disks_75 : ℕ),
              remaining_files_45 = size_45 - used_disks_85 ∧
              used_disks_85 = size_85 ∧
              (remaining_files_45 % 3 = 0 → used_disks_45 = remaining_files_45 / 3) ∧
              (remaining_files_45 % 3 ≠ 0 → used_disks_45 = remaining_files_45 / 3 + 1) ∧
              used_disks_75 = size_75 ∧
              disks = used_disks_85 + used_disks_45 + used_disks_75)) :=
by
  sorry

end min_disks_to_store_files_l508_508951


namespace range_of_f_l508_508888

variable (a b : ℝ)

def f (x : ℝ) : ℝ := a * x + b

theorem range_of_f {a b : ℝ} (ha : a < 0) : set.range (f a b) = set.Icc (a + b) b :=
by
  sorry

end range_of_f_l508_508888


namespace smallest_positive_sigma_l508_508426

-- Given function
def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

-- Shifted function to the left by σ units
def g (x σ : ℝ) : ℝ := f (x + σ)

-- Function to check if g(x) is even
def is_even_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = g (x)

theorem smallest_positive_sigma : ∃ σ : ℝ, 0 < σ ∧ σ = Real.pi / 12 ∧ is_even_function (g σ) :=
  sorry

end smallest_positive_sigma_l508_508426


namespace problem1_l508_508345

theorem problem1 (a b c : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) : 
  a + b + c ≥ Real.sqrt (a * b) + Real.sqrt (b * c) + Real.sqrt (c * a) :=
sorry

end problem1_l508_508345


namespace distance_foci_ellipse_l508_508631

noncomputable def distance_between_foci (x y : ℝ) := 
  (real.sqrt (((2 * x - 6) ^ 2 + (3 * y + 12) ^ 2)) + (real.sqrt (((2 * x + 10) ^ 2 + (3 * y - 24) ^ 2)))) = 30

theorem distance_foci_ellipse : 
  ∀ x y : ℝ, distance_between_foci x y → 
  let f1 := (3 / 2 : ℝ, -4 : ℝ) in
  let f2 := (-5 / 2 : ℝ, 8 : ℝ) in
  real.sqrt(((f1.1 - f2.1)^2 + (f1.2 - f2.2)^2)) = 4 * real.sqrt 10 :=
begin
  sorry
end

end distance_foci_ellipse_l508_508631


namespace op_proof_l508_508890

def op (x y : ℕ) : ℕ := x * y - 2 * x

theorem op_proof : op 7 4 - op 4 7 = -6 := by
  sorry

end op_proof_l508_508890


namespace carla_catches_up_in_three_hours_l508_508957

-- Definitions as lean statements based on conditions
def john_speed : ℝ := 30
def carla_speed : ℝ := 35
def john_start_time : ℝ := 0
def carla_start_time : ℝ := 0.5

-- Lean problem statement to prove the catch-up time
theorem carla_catches_up_in_three_hours : 
  ∃ t : ℝ, 35 * t = 30 * (t + 0.5) ∧ t = 3 :=
by
  sorry

end carla_catches_up_in_three_hours_l508_508957


namespace intersection_of_A_and_B_l508_508105

-- Define the sets A and B given their conditions
def setA := {x : ℝ | ∃ y : ℝ, y = sqrt (x - 1)} 
def setB := {y : ℝ | ∃ x : ℝ, y = -x + 3 ∧ x ∈ setA}

-- Define the statement to be proved: A ∩ B = [1, 2]
theorem intersection_of_A_and_B : {x : ℝ | x ∈ setA ∧ x ∈ setB} = {x : ℝ | 1 ≤ x ∧ x ≤ 2} := 
by 
  sorry

end intersection_of_A_and_B_l508_508105


namespace after_lunch_typing_orders_l508_508933

-- Define the problem conditions
def boss_delivery_order : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9]
def secretary_typing_order (delivered : List ℕ) (typed : List ℕ) : Prop :=
  ∀ l ∈ typed, ∃ idx, idx < delivered.length ∧ delivered[idx] = l ∧ (∀ j, idx < j -> delivered[j] ∉ typed)

-- Define the problem
theorem after_lunch_typing_orders :
  let remaining_letters := [1, 2, 3, 4, 5, 6, 7, 9] in
  (2 ^ remaining_letters.length) = 256 :=
by
  -- Proof steps go here
  sorry

end after_lunch_typing_orders_l508_508933


namespace polynomial_expansion_l508_508783

-- Definitions of the polynomials
def p (w : ℝ) : ℝ := 3 * w^3 + 4 * w^2 - 7
def q (w : ℝ) : ℝ := 2 * w^3 - 3 * w^2 + 1

-- Statement of the theorem
theorem polynomial_expansion (w : ℝ) : 
  (p w) * (q w) = 6 * w^6 - 6 * w^5 + 9 * w^3 + 12 * w^2 - 3 :=
by
  sorry

end polynomial_expansion_l508_508783


namespace distance_least_sqrt_n_l508_508559

-- Define the distances
variables {A B : Type} [metric_space A] [metric_space B]

-- n points P_i
variables {n : ℕ} (P : fin n → A)

-- Distance function
def distance_set (a b : A) (P : fin n → A) : set ℝ :=
  {dist a (P i) | i}

theorem distance_least_sqrt_n
  (P : fin n → A) (hA : ∀ i j : fin n, i ≠ j → dist A (P i) ≠ dist A (P j))
  (hB : ∀ i j : fin n, i ≠ j → dist B (P i) ≠ dist B (P j)) :
  (cardinal.mk (distance_set A P) ∪ (distance_set B P) : ℕ) ≥ int.sqrt n :=
sorry

end distance_least_sqrt_n_l508_508559


namespace A_can_complete_work_in_28_days_l508_508335
noncomputable def work_days_for_A (x : ℕ) (h : 4 / x = 1 / 21) : ℕ :=
  x / 3

theorem A_can_complete_work_in_28_days (x : ℕ) (h : 4 / x = 1 / 21) :
  work_days_for_A x h = 28 :=
  sorry

end A_can_complete_work_in_28_days_l508_508335


namespace simple_interest_rate_l508_508386

theorem simple_interest_rate (P A T : ℝ) (hP : P = 8000) (hA : A = 12500) (hT : T = 7) : 
  ∃ R : ℝ, R = 8.04 ∧ (A - P) = P * R * T / 100 :=
by
  -- Definitions from the problem conditions
  have hSI : A - P = 4500, by  -- Simple Interest
  have hR_formula : 4500 = 8000 * R * 7 / 100, by
  -- Isolate R (Rate)
  have R_isolated : R = 450000 / 56000, by
  have R_approx : R = 8.04, by sorry
  exact ⟨8.04, ⟨R_approx, hR_formula⟩⟩

end simple_interest_rate_l508_508386


namespace find_edge_value_l508_508313

theorem find_edge_value (a b c d e_1 e_2 e_3 e_4 : ℕ) 
  (h1 : e_1 = a + b)
  (h2 : e_2 = b + c)
  (h3 : e_3 = c + d)
  (h4 : e_4 = d + a)
  (h5 : e_1 = 8)
  (h6 : e_3 = 13)
  (h7 : e_1 + e_3 = a + b + c + d)
  : e_4 = 12 := 
by sorry

end find_edge_value_l508_508313


namespace third_place_books_max_l508_508285

theorem third_place_books_max (x y z : ℕ) (hx : 100 ∣ x) (hxpos : 0 < x) (hy : 100 ∣ y) (hz : 100 ∣ z)
  (h_sum : 2 * x + 100 + x + 100 + x + y + z ≤ 10000)
  (h_first_eq : 2 * x + 100 = x + 100 + x)
  (h_second_eq : x + 100 = y + z) 
  : x ≤ 1900 := sorry

end third_place_books_max_l508_508285


namespace find_angle_C_find_max_perimeter_l508_508547

-- Define the first part of the problem
theorem find_angle_C 
  (a b c A B C : ℝ) (h1 : (2 * a + b) * Real.sin A + (2 * b + a) * Real.sin B = 2 * c * Real.sin C) :
  C = (2 * Real.pi) / 3 :=
sorry

-- Define the second part of the problem
theorem find_max_perimeter 
  (a b A B : ℝ)
  (C : ℝ := (2 * Real.pi) / 3)
  (c : ℝ := Real.sqrt 3)
  (h1 : (2 * a + b) * Real.sin A + (2 * b + a) * Real.sin B = 2 * c * Real.sin C) :
  (2 * Real.sqrt 3 < a + b + c) ∧ (a + b + c <= 2 + Real.sqrt 3) :=
sorry

end find_angle_C_find_max_perimeter_l508_508547


namespace length_of_curve_is_correct_l508_508706

noncomputable def length_of_curve {A B C D A₁ B₁ C₁ D₁ : Type*} [metric_space A]
  (cube : set (A × B × C × D × A₁ × B₁ × C₁ × D₁))
  (edge_length : ℝ)
  (P : metric_space A)
  (dist_AP : ℝ) : ℝ :=
if (cube ∈ metric_space.cube A B C D A₁ B₁ C₁ D₁) ∧ (edge_length = 1) ∧ (dist (P, A) = ((2*sqrt 3)/3))
then 5*sqrt 3/6*pi
else 0

-- Statement to prove
theorem length_of_curve_is_correct
  (A B C D A₁ B₁ C₁ D₁ : Type*)
  [metric_space A]
  (cube : set (A × B × C × D × A₁ × B₁ × C₁ × D₁))
  (edge_length : ℝ)
  (P : metric_space A)
  (dist_AP : ℝ) :
  (cube ∈ metric_space.cube A B C D A₁ B₁ C₁ D₁) ∧ (edge_length = 1) ∧ (dist (P, A) = ((2*sqrt 3)/3)) →
  (length_of_curve cube edge_length P dist_AP = 5*sqrt 3/6*pi) :=
begin
  sorry
end

end length_of_curve_is_correct_l508_508706


namespace points_on_circle_l508_508442

theorem points_on_circle (t : ℝ) : 
  ∃ x y : ℝ, (x = Real.cos t + 1) ∧ (y = Real.sin t - 1) ∧ (x - 1)^2 + (y + 1)^2 = 1 :=
by 
  use [Real.cos t + 1, Real.sin t - 1]
  exact ⟨rfl, rfl, by sorry⟩

end points_on_circle_l508_508442


namespace find_YZ_length_l508_508343

variable {Real : Type} [LinearOrderedField Real]

def similar_triangles (AB BC XY : Real) (YZ_correct : Real) : Prop :=
  ∃ (YZ YZ_rounded: Real), 
    (YZ / BC = XY / AB) ∧ 
    (YZ_rounded = Real.floor (YZ * 10) / 10) ∧ 
    (YZ_rounded = YZ_correct)

theorem find_YZ_length :
  similar_triangles 7 10 4 5.7 :=
by
  sorry

end find_YZ_length_l508_508343


namespace arithmetic_sequence_sum_l508_508819

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (d : ℕ) (h1 : a 1 + a 13 = 10) 
  (h2 : ∀ n m : ℕ, a (n + 1) = a n + d) : a 3 + a 5 + a 7 + a 9 + a 11 = 25 :=
  sorry

end arithmetic_sequence_sum_l508_508819


namespace max_projection_area_of_tetrahedron_l508_508307

noncomputable def maxProjectionArea (s: ℝ) (theta: ℝ) : ℝ :=
  let faceArea := (sqrt 3 / 4) * s^2
  faceArea

theorem max_projection_area_of_tetrahedron (s : ℝ) (theta : ℝ) (h0 : s = 3) (h1 : theta = π / 6) :
  maxProjectionArea s theta = (9 * sqrt 3) / 4 :=
by
  sorry

end max_projection_area_of_tetrahedron_l508_508307


namespace least_relative_error_l508_508384

-- Define the conditions
def error1 : ℝ := 0.1
def length1 : ℝ := 20
def error2 : ℝ := 0.03
def length2 : ℝ := 30
def error3 : ℝ := 0.5
def length3 : ℝ := 200

-- Define the relative errors
def relative_error (error length : ℝ) : ℝ := (error / length) * 100

-- Prove the statement
theorem least_relative_error : relative_error error2 length2 = min (relative_error error1 length1) (min (relative_error error2 length2) (relative_error error3 length3)) := 
sorry

end least_relative_error_l508_508384


namespace sum_of_squares_of_sum_and_difference_l508_508644

theorem sum_of_squares_of_sum_and_difference (x y : ℤ) (h1 : x + y = 24) (h2 : x - y = 8) : 
  (x + y)^2 + (x - y)^2 = 640 :=
by
  sorry

end sum_of_squares_of_sum_and_difference_l508_508644


namespace sleeves_in_box_crackers_l508_508760

theorem sleeves_in_box_crackers :
  ∀ (sleeve_capacity : ℕ) (boxes : ℕ) (nights : ℕ) (crackers_per_night : ℕ)
    (total_crackers_consumed : ℕ),
    sleeve_capacity = 28 →
    boxes = 5 →
    nights = 56 →
    crackers_per_night = 10 →
    total_crackers_consumed = crackers_per_night * nights →
    total_crackers_consumed = 560 →
    ((total_crackers_consumed / boxes) / sleeve_capacity) = 4 :=
by
  intros sleeve_capacity boxes nights crackers_per_night total_crackers_consumed
         hsleeve hboxes hnights hcrackers_per_night htotal_cons htcons
  rw hcrackers_per_night at htotal_cons
  rw hnights at htotal_cons
  rw mul_comm at htotal_cons
  simp at htcons
  rw htcons
  rw hboxes
  rw hsleeve
  norm_num
  sorry

end sleeves_in_box_crackers_l508_508760


namespace proof_problem_l508_508616

theorem proof_problem
  (x y : ℤ)
  (hx : ∃ m : ℤ, x = 6 * m)
  (hy : ∃ n : ℤ, y = 12 * n) :
  (x + y) % 2 = 0 ∧ (x + y) % 6 = 0 ∧ ¬ (x + y) % 12 = 0 → ¬ (x + y) % 12 = 0 :=
  sorry

end proof_problem_l508_508616


namespace locus_of_tangent_centers_is_hyperbola_l508_508871
#import Mathlib -- Broader import to cover necessary libraries

-- Define the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 8 * x + 12 = 0

-- Define the condition for a circle being tangent to both circle1 and circle2
def is_tangent (center_x center_y : ℝ) : Prop := 
  ∃ r : ℝ, (center_x - 0)^2 + (center_y - 0)^2 = (r + 1)^2 ∧ (center_x - 4)^2 + center_y^2 = (r + 2)^2

-- Prove that the locus of center of a circle tangent to both circle1 and circle2 forms a hyperbola
theorem locus_of_tangent_centers_is_hyperbola :
  ∃ (locus : ℝ → ℝ → Prop), 
    (∀ center_x center_y, is_tangent center_x center_y → locus center_x center_y) ∧
    (locus = λ x y, ∃ a b : ℝ, (x - a)^2 - (y - b)^2 = 1 ∨ (y - b)^2 - (x - a)^2 = 1) := 
by sorry

end locus_of_tangent_centers_is_hyperbola_l508_508871


namespace divisible_by_exactly_two_l508_508726

theorem divisible_by_exactly_two (S : Set ℕ) (h₁ : S = {n | 1 ≤ n ∧ n ≤ 100})
  (s : ℕ)
  (h_s : s ∈ S)
  (h_div_pairs : ∀ a b c d : ℕ, (a, b, c, d) ∈ {(2, 3, 5, 7)}.powerset 2.toFinset →
                 (∀ x ∈ S, (x % a = 0 ∧ x % b = 0 ∧ x % c ≠ 0 ∧ x % d ≠ 0) ↔ x ∈ t))
: ∑ x in Finset.filter (λ n, ∃ a b, a ≠ b ∧ a ∈ {2, 3, 5, 7} ∧ b ∈ {2, 3, 5, 7} ∧ n % a = 0 ∧ n % b = 0) (Finset.range 101), 1 = 27 := 
sorry

end divisible_by_exactly_two_l508_508726


namespace range_of_x_l508_508508

theorem range_of_x (x : ℝ) : ¬ (x ∈ set.Icc (2 : ℝ) 5 ∨ x < 1 ∨ x > 4) → x ∈ set.Ico 1 2 :=
by
  sorry

end range_of_x_l508_508508


namespace festival_event_ranking_l508_508260

noncomputable def lcm (a b : ℕ) : ℕ := Nat.lcm a b

def Fraction (numerator : ℕ) (denominator : ℕ) := numerator / denominator

theorem festival_event_ranking :
  let dance := Fraction 3 8,
      painting := Fraction 5 16,
      clay_modeling := Fraction 9 24,
      lcm_denominator := lcm 8 (lcm 16 24),
      dance_fraction := (3 * (lcm_denominator / 8)) / lcm_denominator,
      painting_fraction := (5 * (lcm_denominator / 16)) / lcm_denominator,
      clay_modeling_fraction := (9 * (lcm_denominator / 24)) / lcm_denominator in
  dance_fraction = clay_modeling_fraction ∧ 
  dance_fraction > painting_fraction ∧
  clay_modeling_fraction > painting_fraction :=
by
  sorry

end festival_event_ranking_l508_508260


namespace equation_of_line_with_x_intercept_and_slope_l508_508626

theorem equation_of_line_with_x_intercept_and_slope :
  ∃ (a b c : ℝ), a * x - b * y + c = 0 ∧ a = 1 ∧ b = -1 ∧ c = 2 :=
sorry

end equation_of_line_with_x_intercept_and_slope_l508_508626


namespace angle_C_measure_l508_508234

theorem angle_C_measure :
  ∀ (p q r : Line) (A B C : Angle), 
  (parallel p q) → 
  (measure_angle A = (1 / 6) * measure_angle B) → 
  (∃ x : Real, measure_angle C = 180 - 6 * x ∧ x = 180 / 7) :=
by
  sorry

end angle_C_measure_l508_508234


namespace total_holiday_savings_l508_508310

-- Definitions of the conditions
def Victory_saves_less (Sam_savings: ℕ) : ℕ := Sam_savings - 100
def Sam_savings : ℕ := 1000

-- Statement to prove
theorem total_holiday_savings : 
  let Victory_savings := Victory_saves_less Sam_savings in
  Sam_savings + Victory_savings = 1900 := 
by
  sorry

end total_holiday_savings_l508_508310


namespace smallest_positive_period_interval_of_monotonic_increase_value_of_a_l508_508231

-- Problem 1: Smallest positive period of f(x)
noncomputable def f (x : ℝ) : ℝ := 2 * real.sin (2 * x  +  (real.pi / 6)) + 1

theorem smallest_positive_period : 
  ∃ T > 0, ∀ x, f (x + T) = f x :=
sorry

-- Problem 2: Interval of monotonic increase of f(x)
theorem interval_of_monotonic_increase :
  ∀ k : ℤ, 
  ∃ a b : ℝ, ∀ x ∈ set.Icc a b, 
  f' x > 0 ∧ a = - real.pi / 3 + k * real.pi ∧ b = real.pi / 6 + k * real.pi :=
sorry

-- Problem 3: Value of a in triangle ABC given conditions
variables (b c A : ℝ)
def triangle_area (b c A : ℝ) : ℝ := 1 / 2 * b * c * real.sin A

-- Given conditions
axiom f_A : f A = 2
axiom b_eq : b = 1
axiom area_eq : triangle_area 1 c A = real.sqrt 3

-- Prove that a^2 = 13
theorem value_of_a :
  ∃ a : ℝ, a ^ 2 = 13 :=
sorry

end smallest_positive_period_interval_of_monotonic_increase_value_of_a_l508_508231


namespace steven_apples_peaches_difference_l508_508550

def steven_apples := 19
def jake_apples (steven_apples : ℕ) := steven_apples + 4
def jake_peaches (steven_peaches : ℕ) := steven_peaches - 3

theorem steven_apples_peaches_difference (P : ℕ) :
  19 - P = steven_apples - P :=
by
  sorry

end steven_apples_peaches_difference_l508_508550


namespace dot_product_is_correct_l508_508791

theorem dot_product_is_correct : 
    let v1 := (4 : ℤ, -5, 2, -1)
    let v2 := (-6 : ℤ, 3, -4, 2)
    v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3 + v1.4 * v2.4 = -49 := 
by 
  let v1 := (4, -5, 2, -1)
  let v2 := (-6, 3, -4, 2)
  sorry

end dot_product_is_correct_l508_508791


namespace repeating_decimal_periodic_l508_508597

theorem repeating_decimal_periodic (n : ℕ) :
  (∃ m : ℕ, m = 2 * n + 1 ∧ ((1 : ℚ)/m).periodic_period = 6 ∧ sum_of_periodic_digits ((1 : ℚ)/m) = 999)
  ↔ (2 * n + 1 = 7 ∨ 2 * n + 1 = 13) :=
by sorry

end repeating_decimal_periodic_l508_508597


namespace max_FM_l508_508898

noncomputable def maxF (M : ℕ) : ℕ :=
  let a := M / 1000
  let b := (M % 1000) / 100
  let c := (M % 100) / 10
  let d := M % 10
  if b + c + d = 12 ∧ a = b - d ∧ (F(M) / 9).isInt then
    9
  else
    sorry -- other cases

-- Definitions of M, N, F
def F (M : ℕ) : ℤ :=
  let a := M / 1000
  let b := (M % 1000) / 100
  let c := (M % 100) / 10
  let d := M % 10
  let N := 1000 * b + 100 * a + 10 * d + c
  (M - N : ℤ) / 9

-- The problem statement in Lean 4
theorem max_FM (M : ℕ) (h1 : ∃ a b c d : ℕ, M = 1000 * a + 100 * b + 10 * c + d ∧ b + c + d = 12 ∧ a = b - d)
  (h2 : (F M / 9).isInt) : maxF M = 9 := by
  sorry

end max_FM_l508_508898


namespace discarded_number_l508_508621

theorem discarded_number (S x : ℕ) (h1 : S / 50 = 50) (h2 : (S - x - 55) / 48 = 50) : x = 45 :=
by
  sorry

end discarded_number_l508_508621


namespace largest_n_divides_l508_508882

theorem largest_n_divides (n : ℕ) (h : 2^n ∣ 5^256 - 1) : n ≤ 10 := sorry

end largest_n_divides_l508_508882


namespace product_of_solutions_is_zero_l508_508640

theorem product_of_solutions_is_zero :
  (∀ x : ℝ, ((x + 3) / (3 * x + 3) = (5 * x + 4) / (8 * x + 4) -> x = 0)) -> true :=
by
  sorry

end product_of_solutions_is_zero_l508_508640


namespace exists_c1_c2_lambda_l508_508439

def n_sequence (n : ℕ) (a : ℕ → ℕ) : Prop :=
  ∀ i j : ℕ, i + j ≤ n → a i + a j ≤ n ∧ a (a i + a j) = a (i + j)

noncomputable def f (n : ℕ) : ℕ := sorry

theorem exists_c1_c2_lambda :
  ∃ (c1 c2 : ℝ) (λ : ℝ), (0 < c1) ∧ (0 < c2) ∧ (0 < λ) ∧ (λ = 3^(1/6)) ∧
  ∀ n : ℕ, c1 * λ^n < (f n : ℝ) ∧ (f n : ℝ) < c2 * λ^n := 
  sorry

end exists_c1_c2_lambda_l508_508439


namespace P_leq_Q_l508_508229

variables {a b c d m n : ℝ}
-- All variables are positive real numbers.
-- Definitions of P and Q
def P := sqrt (a * b) + sqrt (c * d)
def Q := sqrt (m * a + n * c) * sqrt ((b / m) + (d / n))

-- The proposition that needs to be proved.
theorem P_leq_Q (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (hm : 0 < m) (hn : 0 < n) : P a b c d ≤ Q a b c d :=
by
  sorry

end P_leq_Q_l508_508229


namespace patricia_earns_more_than_jose_l508_508960

noncomputable def jose_final_amount : ℝ :=
  50000 * (1 + 0.04)^2

noncomputable def patricia_final_amount : ℝ :=
  50000 * (1 + 0.01)^8

theorem patricia_earns_more_than_jose :
  patricia_final_amount - jose_final_amount = 63 :=
by
  -- from solution steps
  /-
  jose_final_amount = 50000 * (1 + 0.04)^2 = 54080
  patricia_final_amount = 50000 * (1 + 0.01)^8 ≈ 54143
  patricia_final_amount - jose_final_amount ≈ 63
  -/
  sorry

end patricia_earns_more_than_jose_l508_508960


namespace rhombus_area_l508_508114

-- Define the lengths of the diagonals
def d1 : ℝ := 6
def d2 : ℝ := 8

-- Problem statement: The area of the rhombus
theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 6) (h2 : d2 = 8) : (1 / 2) * d1 * d2 = 24 := by
  -- The proof is not required, so we use sorry.
  sorry

end rhombus_area_l508_508114


namespace midpoint_trajectory_of_moving_point_l508_508452

/-- Given a fixed point A (4, -3) and a moving point B on the circle (x+1)^2 + y^2 = 4, prove that 
    the equation of the trajectory of the midpoint M of the line segment AB is 
    (x - 3/2)^2 + (y + 3/2)^2 = 1. -/
theorem midpoint_trajectory_of_moving_point {x y : ℝ} :
  (∃ (B : ℝ × ℝ), (B.1 + 1)^2 + B.2^2 = 4 ∧ 
    (x, y) = ((B.1 + 4) / 2, (B.2 - 3) / 2)) →
  (x - 3/2)^2 + (y + 3/2)^2 = 1 :=
by sorry

end midpoint_trajectory_of_moving_point_l508_508452


namespace range_of_k_l508_508132

theorem range_of_k {k : ℝ} : (∀ x : ℝ, x < 0 → (k - 2)/x > 0) ∧ (∀ x : ℝ, x > 0 → (k - 2)/x < 0) → k < 2 := 
by
  sorry

end range_of_k_l508_508132


namespace product_of_midpoint_coords_eq_neg24_l508_508318

-- Definition of endpoints
def point1 : ℝ × ℝ := (3, -4)
def point2 : ℝ × ℝ := (5, -8)

-- Midpoint calculation
noncomputable def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ := 
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

-- Product of the coordinates of the midpoint
noncomputable def product_of_coordinates (p : ℝ × ℝ) : ℝ :=
  p.1 * p.2

theorem product_of_midpoint_coords_eq_neg24 :
  product_of_coordinates (midpoint point1 point2) = -24 := 
sorry

end product_of_midpoint_coords_eq_neg24_l508_508318


namespace evaluated_sum_approx_2024_l508_508059

def numerator := (∑ i in Finset.range (2023 + 1), (2024 - i) / i)
def denominator := (∑ i in Finset.range (2024 - 1), 1 / (i + 2))

theorem evaluated_sum_approx_2024 :
  (numerator / denominator) = 2024 - (1 / denominator) :=
by { sorry }

end evaluated_sum_approx_2024_l508_508059


namespace tangent_line_through_points_of_tangency_l508_508097

noncomputable def equation_of_tangent_line (x1 y1 x y : ℝ) : Prop :=
x1 * x + (y1 - 2) * (y - 2) = 4

theorem tangent_line_through_points_of_tangency
  (x1 y1 x2 y2 : ℝ)
  (h1 : equation_of_tangent_line x1 y1 2 (-2))
  (h2 : equation_of_tangent_line x2 y2 2 (-2)) :
  (2 * x1 - 4 * (y1 - 2) = 4) ∧ (2 * x2 - 4 * (y2 - 2) = 4) →
  ∃ a b c, (a = 1) ∧ (b = -2) ∧ (c = 2) ∧ (a * x + b * y + c = 0) :=
by
  sorry

end tangent_line_through_points_of_tangency_l508_508097


namespace min_lambda_l508_508828

open Real

-- Definitions of the properties of the parabola and focus
def parabola_eq (x : ℝ) : ℝ := 4 * x^2
def focus : ℝ × ℝ := (0, 1/16)
def directrix : ℝ := -1/16

-- Definitions for points M and N on the parabola
def M (x : ℝ) : ℝ × ℝ := (x, parabola_eq x)
def N (y : ℝ) : ℝ × ℝ := (y, parabola_eq y)

-- Given angle condition between M, F, and N
def angle_MFN : ℝ := 135

-- Midpoint calculation
def midpoint_chord (M N : ℝ × ℝ) : ℝ × ℝ := ((M.1 + N.1) / 2, (M.2 + N.2) / 2)

-- Distance from midpoint P to the directrix line
def distance_to_directrix (P : ℝ × ℝ) : ℝ := abs (P.2 - directrix)

-- The general problem of finding the minimum value of λ
theorem min_lambda (x y : ℝ) (h_angle : angle_MFN = 135) :
  ∃ λ, λ = 2 + sqrt 2 ∧ (dist (M x) (N y))^2 = λ * (distance_to_directrix (midpoint_chord (M x) (N y)))^2 := sorry

end min_lambda_l508_508828


namespace seq_converges_to_zero_l508_508135

def sequence (x : ℕ → ℝ) : Prop :=
  x 1 = 1 ∧ x 2 = -1 ∧ ∀ n ≥ 1, x (n + 2) = (x (n + 1))^2 - (1 / 2) * x n

theorem seq_converges_to_zero (x : ℕ → ℝ) (h : sequence x) : 
  filter.tendsto x filter.at_top (𝓝 0) :=
sorry

end seq_converges_to_zero_l508_508135


namespace extreme_value_point_l508_508629

def f (x : ℝ) : ℝ := x - 2 * Real.log x

theorem extreme_value_point :
  ∃ x : ℝ, 0 < x ∧ (∀ y : ℝ, 0 < y ∧ y < x → f y < f x) ∧ (∀ y : ℝ, x < y → f y < f x) ∧ x = 2 :=
by
  sorry

end extreme_value_point_l508_508629


namespace series_value_l508_508777

noncomputable def alternating_sum : ℕ → ℤ
| 1 := -1
| 2 := 2 + alternating_sum 1
| n := if ∃ k: ℕ, k^2 ≤ n ∧ n < (k+1)^2
       then if (k % 2 = 1)
            then -(n + alternating_sum (n-1))
            else n + alternating_sum (n-1)
       else alternating_sum (n-1)

theorem series_value : alternating_sum 64 = -1808 :=
by
  sorry

end series_value_l508_508777


namespace pencils_per_row_l508_508418

theorem pencils_per_row (p r : ℕ) (hpp : p = 154) (hpr : r = 14) : p / r = 11 :=
by
  rw [hpp, hpr]
  norm_num
  sorry

end pencils_per_row_l508_508418


namespace kimberly_initial_skittles_l508_508966

theorem kimberly_initial_skittles : 
  ∀ (x : ℕ), (x + 7 = 12) → x = 5 :=
by
  sorry

end kimberly_initial_skittles_l508_508966


namespace simplify_expression_l508_508394

variable (x y : ℝ)

theorem simplify_expression : (x^2 + x * y) / (x * y) * (y^2 / (x + y)) = y := by
  sorry

end simplify_expression_l508_508394


namespace blake_change_l508_508748

def cost_oranges : ℕ := 40
def cost_apples : ℕ := 50
def cost_mangoes : ℕ := 60
def initial_money : ℕ := 300

def total_cost : ℕ := cost_oranges + cost_apples + cost_mangoes
def change : ℕ := initial_money - total_cost

theorem blake_change : change = 150 := by
  sorry

end blake_change_l508_508748


namespace part_a_part_b_part_c_part_d_l508_508529

variables (City : Type) (airline railway : City → City → Prop)

-- Conditions: Every two cities are connected by either airline or railway
axiom connectivity : ∀ (a b : City), airline a b ∨ railway a b

-- a) It is possible to choose a mode of transportation such that one can travel from any city to any other city using only this mode of transportation.
theorem part_a : ∃ (mode : City → City → Prop), (∀ (a b : City), ∃ (p : List City), List.chain' mode p ∧ List.head! p = a ∧ List.last p sorry = b) :=
sorry

-- b) From a certain city, choosing one mode of transportation, one can travel to any other city with no more than one transfer (only using the chosen mode of transportation).
theorem part_b : ∃ (mode : City → City → Prop), ∀ (a b : City), (∃ (x : City), mode a x ∧ mode x b) ∨ mode a b :=
sorry

-- c) Each city possesses the property described in b).
theorem part_c : ∀ (a : City), ∃ (mode : City → City → Prop), ∀ (b : City), (∃ (x : City), mode a x ∧ mode x b) ∨ mode a b :=
sorry

-- d) It is possible to choose a mode of transportation such that using only that mode, one can travel from any city to any other city with no more than two transfers.
theorem part_d : ∃ (mode : City → City → Prop), ∀ (a b : City), 
  (∃ (x y : City), mode a x ∧ mode x y ∧ mode y b) ∨ (∃ (x : City), mode a x ∧ mode x b) ∨ mode a b :=
sorry

end part_a_part_b_part_c_part_d_l508_508529


namespace seating_arrangement_l508_508654

noncomputable def factorial : ℕ → ℕ
  | 0       => 1
  | (n + 1) => (n + 1) * factorial n

theorem seating_arrangement :
  let n := 6 in
  let factorial_n := factorial n in
  2 * (factorial_n * factorial_n) = 1036800 := by
  sorry

end seating_arrangement_l508_508654


namespace eternal_number_max_FM_l508_508901

theorem eternal_number_max_FM
  (a b c d : ℕ)
  (h1 : b + c + d = 12)
  (h2 : a = b - d)
  (h3 : (1000 * a + 100 * b + 10 * c + d) - (1000 * b + 100 * a + 10 * d + c) = 81 * (100 * a - 100 * b + c - d))
  (h4 : ∃ k : ℤ, F(M) = 9 * k) :
  ∃ a b c d : ℕ, 100 * (b - d) - 100 * b + 12 - b - 102 * d = 9 := sorry

end eternal_number_max_FM_l508_508901


namespace union_A_B_complement_U_A_intersection_B_range_of_a_l508_508834

-- Define the sets A, B, C, and U
def setA (x : ℝ) : Prop := 2 ≤ x ∧ x ≤ 8
def setB (x : ℝ) : Prop := 1 < x ∧ x < 6
def setC (a : ℝ) (x : ℝ) : Prop := x > a
def U (x : ℝ) : Prop := True  -- U being the universal set of all real numbers

-- Define complements and intersections
def complement (A : ℝ → Prop) (x : ℝ) : Prop := ¬ A x
def intersection (A B : ℝ → Prop) (x : ℝ) : Prop := A x ∧ B x
def union (A B : ℝ → Prop) (x : ℝ) : Prop := A x ∨ B x

-- Proof problems
theorem union_A_B : ∀ x, union setA setB x ↔ (1 < x ∧ x ≤ 8) :=
by 
  intros x
  sorry

theorem complement_U_A_intersection_B : ∀ x, intersection (complement setA) setB x ↔ (1 < x ∧ x < 2) :=
by 
  intros x
  sorry

theorem range_of_a (a : ℝ) : (∃ x, intersection setA (setC a) x) → a < 8 :=
by
  intros h
  sorry

end union_A_B_complement_U_A_intersection_B_range_of_a_l508_508834


namespace line_is_tangent_to_circle_l508_508522

theorem line_is_tangent_to_circle
  (θ : Real)
  (l : ℝ → ℝ → Prop) (C : ℝ → ℝ → Prop)
  (h_l : ∀ x y, l x y ↔ x * Real.sin θ + 2 * y * Real.cos θ = 1)
  (h_C : ∀ x y, C x y ↔ x^2 + y^2 = 1) :
  (∀ x y, l x y ↔ x = 1 ∨ x = -1) ↔
  (∃ x y, C x y ∧ ∀ x y, l x y → Real.sqrt ((x * Real.sin θ + 2 * y * Real.cos θ - 1)^2 / (Real.sin θ^2 + 4 * Real.cos θ^2)) = 1) :=
sorry

end line_is_tangent_to_circle_l508_508522


namespace num_roots_l508_508280

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 3 * x - 2

theorem num_roots : ∃! x : ℝ, f x = 0 := 
sorry

end num_roots_l508_508280


namespace circle_tangent_count_l508_508648

-- Define the basic properties and circles
structure Circle :=
(center : ℝ × ℝ)
(radius : ℝ)

-- Define given circles
def C1 : Circle := { center := (0, 0), radius := 2 }
def C2 : Circle := { center := (4, 0), radius := 2 }
def C3 : Circle := { center := (2, 2 * Real.sqrt 3), radius := 2 }

-- Predicate for a circle being tangent to another circle
def tangent (C D : Circle) : Prop :=
  Real.dist C.center D.center = C.radius + D.radius

-- Predicate for a circle being a valid solution
def valid_circle (D : Circle) : Prop :=
  D.radius = 4 ∧ 
  tangent D C1 ∧ 
  tangent D C2 ∧ 
  tangent D C3

-- Statement of the proof problem
theorem circle_tangent_count : ∃! (D1 D2 : Circle), valid_circle D1 ∧ valid_circle D2 :=
by 
  -- The proof is omitted, using sorry to indicate no proof is provided
  sorry

end circle_tangent_count_l508_508648


namespace dave_final_tickets_l508_508740

variable (initial_tickets_set1_won : ℕ) (initial_tickets_set1_lost : ℕ)
variable (initial_tickets_set2_won : ℕ) (initial_tickets_set2_lost : ℕ)
variable (multiplier_set3 : ℕ)
variable (initial_tickets_set3_lost : ℕ)
variable (used_tickets : ℕ)
variable (additional_tickets : ℕ)

theorem dave_final_tickets :
  let net_gain_set1 := initial_tickets_set1_won - initial_tickets_set1_lost
  let net_gain_set2 := initial_tickets_set2_won - initial_tickets_set2_lost
  let net_gain_set3 := multiplier_set3 * net_gain_set1 - initial_tickets_set3_lost
  let total_tickets_after_sets := net_gain_set1 + net_gain_set2 + net_gain_set3
  let tickets_after_buying := total_tickets_after_sets - used_tickets
  let final_tickets := tickets_after_buying + additional_tickets
  initial_tickets_set1_won = 14 →
  initial_tickets_set1_lost = 2 →
  initial_tickets_set2_won = 8 →
  initial_tickets_set2_lost = 5 →
  multiplier_set3 = 3 →
  initial_tickets_set3_lost = 15 →
  used_tickets = 25 →
  additional_tickets = 7 →
  final_tickets = 18 :=
by
  intros
  sorry

end dave_final_tickets_l508_508740


namespace mutually_prime_sum_l508_508169

open Real

theorem mutually_prime_sum (A B C : ℤ) (h_prime : Int.gcd A (Int.gcd B C) = 1)
    (h_eq : A * log 5 / log 200 + B * log 2 / log 200 = C) : A + B + C = 6 := 
sorry

end mutually_prime_sum_l508_508169


namespace six_by_six_tiled_by_dominoes_l508_508348

theorem six_by_six_tiled_by_dominoes (h : ∃ t : ℕ × ℕ, t = (6, 6) ∧ t.snd = 18) : 
  ∃ l : ℕ × ℕ, l.fst = 3 ∨ l.snd = 3 :=
begin
  sorry
end

end six_by_six_tiled_by_dominoes_l508_508348


namespace perpendicular_vectors_x_value_l508_508142

theorem perpendicular_vectors_x_value:
  ∀ (x : ℝ), let a : ℝ × ℝ := (1, 2)
             let b : ℝ × ℝ := (x, 1)
             (a.1 * b.1 + a.2 * b.2 = 0) → x = -2 :=
by
  intro x
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (x, 1)
  intro h
  sorry

end perpendicular_vectors_x_value_l508_508142


namespace Pi_2011_val_l508_508230

section
variables (a : ℕ → ℝ)
-- Condition 1: initial term of the sequence
axiom a1 : a 1 = 2
-- Condition 2: recursive definition of the sequence
axiom recurrence : ∀ n : ℕ, n ≥ 1 → a (n + 1) = 1 / a n
-- Definition of the product of the first n terms
noncomputable def Π_n (n : ℕ) : ℝ := ∏ i in Finset.range n, a (i + 1)

-- Theorem to prove
theorem Pi_2011_val : Π_n a 2011 = 2 :=
sorry
end

end Pi_2011_val_l508_508230


namespace max_sides_13_eq_13_max_sides_1950_eq_1950_l508_508928

noncomputable def max_sides (n : ℕ) : ℕ := n

theorem max_sides_13_eq_13 : max_sides 13 = 13 :=
by {
  sorry
}

theorem max_sides_1950_eq_1950 : max_sides 1950 = 1950 :=
by {
  sorry
}

end max_sides_13_eq_13_max_sides_1950_eq_1950_l508_508928


namespace range_of_m_l508_508884

theorem range_of_m 
  (h : ∀ x, -1 < x ∧ x < 4 → x > 2 * (m: ℝ)^2 - 3)
  : ∀ (m: ℝ), -1 ≤ m ∧ m ≤ 1 :=
by 
  sorry

end range_of_m_l508_508884


namespace divide_into_three_provinces_l508_508339

-- Definitions of the conditions
structure City := (id : Nat)
structure Road := (start end : City) (one_way : Bool)

def Country := List City
def Roads := List Road

-- The critical condition: there's a unique path between any two distinct cities without re-visiting any city
def unique_path_condition (roads : Roads) : Prop :=
  ∀ (c1 c2 : City), c1 ≠ c2 → ∃! (p : List Road), (p.head.start = c1) ∧ (p.last.end = c2) ∧ (∀ (r : Road) ∈ p, r ∉ p.rest)

-- The final theorem statement to be proved
theorem divide_into_three_provinces (C : Country) (R : Roads) (H : unique_path_condition R) :
  ∃ (P1 P2 P3 : List City), 
  P1 ∪ P2 ∪ P3 = C ∧ 
  ∀ (r : Road), (r.start ∈ P1 ∧ r.end ∈ P1) → False ∧ 
                 (r.start ∈ P2 ∧ r.end ∈ P2) → False ∧ 
                 (r.start ∈ P3 ∧ r.end ∈ P3) → False ∧ 
                 (r.start ∈ P1 ∧ r.end ∈ P2) → True ∧
                 (r.start ∈ P2 ∧ r.end ∈ P3) → True ∧
                 (r.start ∈ P3 ∧ r.end ∈ P1) → True := 
sorry

end divide_into_three_provinces_l508_508339


namespace exists_zero_in_interval_l508_508634

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + sin x - 2 * x

-- Expressing in Lean 4 the existence of zero in the interval (1, π/2)
theorem exists_zero_in_interval : ∃ x ∈ Ioo 1 (Real.pi / 2), f x = 0 := by
  sorry

end exists_zero_in_interval_l508_508634


namespace part1_union_part1_intersect_complement_part2_range_a_l508_508831

open Set

-- Problem Setup
namespace ProofProblem

variables (x a : ℝ) (U : Set ℝ := univ)
def A := {x : ℝ | 2 ≤ x ∧ x ≤ 8}
def B := {x : ℝ | 1 < x ∧ x < 6}
def C (a : ℝ) := {x : ℝ | x > a}

-- Proof Statements

theorem part1_union : A ∪ B = {x : ℝ | 1 < x ∧ x ≤ 8} := 
by sorry

theorem part1_intersect_complement : (U \ A) ∩ B = {x : ℝ | 1 < x ∧ x < 2} :=
by sorry

theorem part2_range_a (h : (A ∩ C a) ≠ ∅) : a < 8 :=
by sorry

end ProofProblem

end part1_union_part1_intersect_complement_part2_range_a_l508_508831


namespace pentagon_regular_l508_508541

def isRegularPentagon (ABCDE : List Point) : Prop :=
  let A := ABCDE[0]
  let B := ABCDE[1]
  let C := ABCDE[2]
  let D := ABCDE[3]
  let E := ABCDE[4]
  (dist B C = dist C D ∧ dist C D = dist D E) ∧
  (isParallel (line A C) (line D E) ∧
   isParallel (line B D) (line A E) ∧
   isParallel (line A D) (line B E) ∧
   isParallel (line A B) (line C E) ∧
   isParallel (line C D) (line A E)) →
  (dist A B = dist B C ∧ dist B C = dist C D ∧
   dist C D = dist D E ∧ dist D E = dist E A) ∧
  (∠A = ∠B ∧ ∠B = ∠C ∧ ∠C = ∠D ∧ ∠D = ∠E)

theorem pentagon_regular
  {A B C D E : Point}
  (h1 : dist B C = dist C D)
  (h2 : dist C D = dist D E)
  (h3 : isParallel (line A C) (line D E))
  (h4 : isParallel (line B D) (line A E))
  (h5 : isParallel (line A D) (line B E))
  (h6 : isParallel (line A B) (line C E))
  (h7 : isParallel (line C D) (line A E)) :
  isRegularPentagon [A, B, C, D, E] :=
sorry

end pentagon_regular_l508_508541


namespace possible_scores_count_l508_508533

theorem possible_scores_count (x y : ℕ) (h : x + y = 7) (hx : ∀ k, k ≤ x → k = 1) (hy : ∀ k, k ≤ y → k = 3) : 
  {score | ∃ i j, (i + j = 7) ∧ (j * 1 + i * 3 = score)}.card = 8 := 
sorry

end possible_scores_count_l508_508533


namespace total_cases_sold_l508_508299

theorem total_cases_sold : 
  let people := 20 in
  let first_8_cases := 8 * 3 in
  let next_4_cases := 4 * 2 in
  let last_8_cases := 8 * 1 in
  first_8_cases + next_4_cases + last_8_cases = 40 := 
by
  let people := 20
  let first_8_cases := 8 * 3
  let next_4_cases := 4 * 2
  let last_8_cases := 8 * 1
  have h1 : first_8_cases = 24 := by rfl
  have h2 : next_4_cases = 8 := by rfl
  have h3 : last_8_cases = 8 := by rfl
  have h : first_8_cases + next_4_cases + last_8_cases = 24 + 8 + 8 := by rw [h1, h2, h3]
  show 24 + 8 + 8 = 40 from rfl

end total_cases_sold_l508_508299


namespace sodium_hydroxide_formed_l508_508424

-- Define the entities involved in the chemical reaction
def NaH : Type := Unit
def H2O : Type := Unit
def NaOH : Type := Unit
def H2 : Type := Unit

-- Define the balanced chemical reaction
noncomputable def balanced_reaction (n_NaH n_H2O : ℕ) : ℕ × ℕ :=
  if n_NaH = 1 ∧ n_H2O = 1 then (1, 1) else (0, 0)

-- Define the chemical reaction properties
theorem sodium_hydroxide_formed (n_NaH n_H2O : ℕ) :
  (balanced_reaction n_NaH n_H2O).1 = 1 → ∃ n_NaOH : ℕ, n_NaOH = 1 :=
by {
  intro h,
  existsi 1,
  assumption
}

end sodium_hydroxide_formed_l508_508424


namespace correct_avg_marks_l508_508669

-- Problem conditions
def avg_marks := 100
def num_students := 30
def wrong_mark := 70
def correct_mark := 10

-- Goal
theorem correct_avg_marks :
  let incorrect_total_marks := avg_marks * num_students in
  let correct_total_marks := incorrect_total_marks - wrong_mark + correct_mark in
  correct_total_marks / num_students = 98 :=
by
  let incorrect_total_marks := avg_marks * num_students
  let correct_total_marks := incorrect_total_marks - wrong_mark + correct_mark
  show correct_total_marks / num_students = 98
  sorry -- Proof not provided

end correct_avg_marks_l508_508669


namespace symmetric_line_and_distance_l508_508134

-- Define the lines l1 and l2
def l1_equation (x y : ℝ) : Prop := 2 * x + y + 3 = 0
def l2_equation (x y : ℝ) : Prop := x - 2 * y = 0

-- Prove the statement with the given conditions
theorem symmetric_line_and_distance :
  (∀ (x y : ℝ), (l1_equation x y ↔ 2 * x + y + 3 = 0) ∧
                (l2_equation x y ↔ x - 2 * y = 0)) →
  (∃ (x3 y3 : ℝ), (2 * x3 - y3 + 3 = 0) ∧ (x3 = -2) ∧ (y3 = -1)) ∧
  ((3 * -2 + 4 * -1 + 10 = 0) ∨ (-2 = -2)) :=
by
  intros,
  sorry

end symmetric_line_and_distance_l508_508134


namespace combined_weight_of_boxes_l508_508554

def first_box_weight := 2
def second_box_weight := 11
def last_box_weight := 5

theorem combined_weight_of_boxes :
  first_box_weight + second_box_weight + last_box_weight = 18 := by
  sorry

end combined_weight_of_boxes_l508_508554


namespace max_cities_l508_508530

theorem max_cities (n : ℕ) (h1 : ∀ (c : Fin n), ∃ (neighbors : Finset (Fin n)), neighbors.card ≤ 3 ∧ c ∈ neighbors) (h2 : ∀ (c1 c2 : Fin n), c1 ≠ c2 → ∃ c : Fin n, c1 ≠ c ∧ c2 ≠ c) : n ≤ 10 := 
sorry

end max_cities_l508_508530


namespace geese_population_1996_l508_508637

theorem geese_population_1996 (k x : ℝ) 
  (h1 : x - 39 = k * 60) 
  (h2 : 123 - 60 = k * x) : 
  x = 84 := 
by
  sorry

end geese_population_1996_l508_508637


namespace opposite_numbers_expression_l508_508886

theorem opposite_numbers_expression (a b : ℤ) (h : a + b = 0) : 3 * a + 3 * b - 2 = -2 :=
by
  sorry

end opposite_numbers_expression_l508_508886


namespace price_first_day_l508_508246

variables (O : ℝ) (P1 P2 : ℝ)
-- Conditions
def condition1 (h1 : P2 = 0.40)
(def condition2 (h2 : 2 * O * P1 = 3 * O * P2)

-- Assertion
theorem price_first_day (h1 : P2 = 0.40) (h2 : 2 * O * P1 = 3 * O * P2) : P1 = 0.60 :=
by
  sorry

end price_first_day_l508_508246


namespace binomial_third_term_constant_l508_508525

theorem binomial_third_term_constant (n : ℕ) (T_3_is_constant : 
  ∃ (k : ℕ), binomial k n * (3:ℤ)^k * (-2:ℤ)^(n-k) * (x^(k - (n-k)) = c) (constant_term : T_3_is_constant -> ∃ c, x^c = 1) :
  n = 8 :=
by
  sorry

end binomial_third_term_constant_l508_508525


namespace Yoongi_score_is_53_l508_508070

-- Define the scores of the three students
variables (score_Yoongi score_Eunji score_Yuna : ℕ)

-- Define the conditions given in the problem
axiom Yoongi_Eunji : score_Eunji = score_Yoongi - 25
axiom Eunji_Yuna  : score_Yuna = score_Eunji - 20
axiom Yuna_score  : score_Yuna = 8

theorem Yoongi_score_is_53 : score_Yoongi = 53 := by
  sorry

end Yoongi_score_is_53_l508_508070


namespace k_range_quadrants_l508_508113

noncomputable def k_range_valid (k : ℝ) : Prop :=
  let y := λ x:ℝ, k * x + (2 - k) in
  0 < k ∧ k < 2

theorem k_range_quadrants (k : ℝ) :
  (∃ (y : ℝ → ℝ), y = λ x:ℝ, k * x + (2 - k) ∧ (∃ x1 x2 x3, x1 > 0 ∧ y x1 > 0 ∧ x2 < 0 ∧ y x2 > 0 ∧ x3 > 0 ∧ y x3 < 0)) →
  k_range_valid k :=
by
  sorry

end k_range_quadrants_l508_508113


namespace blake_change_given_l508_508751

theorem blake_change_given :
  let oranges := 40
  let apples := 50
  let mangoes := 60
  let total_amount := 300
  let total_spent := oranges + apples + mangoes
  let change_given := total_amount - total_spent
  change_given = 150 :=
by
  sorry

end blake_change_given_l508_508751


namespace scientific_notation_11580000_l508_508911

theorem scientific_notation_11580000 :
  (11580000 : ℝ) = 1.158 * 10^7 :=
sorry

end scientific_notation_11580000_l508_508911


namespace ratio_of_constants_l508_508253

theorem ratio_of_constants (
  c r s : ℝ
  (h : 8 * k^2 - 12 * k + 20 = c * (k + r)^2 + s)
) : s / r = -62 / 3 :=
sorry

end ratio_of_constants_l508_508253


namespace smallest_sum_of_five_consecutive_odds_l508_508325

theorem smallest_sum_of_five_consecutive_odds :
  35 = min {n ∈ {35, 65, 125, 145, 175} | ∃ k, n = k + (k + 2) + (k + 4) + (k + 6) + (k + 8)} :=
by
  sorry

end smallest_sum_of_five_consecutive_odds_l508_508325


namespace opposite_edges_perpendicular_l508_508680

variables {A B C D E F G H I J: Type*} [LinearOrderedField A] [EuclideanGeometry A]

structure Tetrahedron (V : Type*) :=
(A B C D : V)

structure Midpoints (V : Type*) :=
(E F G H I J : V)

-- Define a predicate for equal distances between midpoints
def distances_equal (E F G H I J : A) : Prop :=
  dist E F = dist G H ∧ dist G H = dist I J

-- Prove the opposite edges of a tetrahedron are perpendicular.
theorem opposite_edges_perpendicular (tet: Tetrahedron A) (mid: Midpoints A)
  (h : distances_equal mid.E mid.F mid.G mid.H mid.I mid.J):
  (opposite_edges_perpendicular_property tet) := 
sorry

end opposite_edges_perpendicular_l508_508680


namespace _l508_508738

axiom abacus_theorem :
  ∃ (xyz abc : ℕ), 
    100 ≤ xyz ∧ xyz < 1000 ∧
    100 ≤ abc ∧ abc < 1000 ∧
    let digits_distinct (n : ℕ) := 
      let d1 := n / 100
      let d2 := (n / 10) % 10
      let d3 := n % 10
      d1 ≠ d2 ∧ d2 ≠ d3 ∧ d1 ≠ d3
    in digits_distinct xyz ∧ digits_distinct abc ∧
    xyz % abc = 0 ∧
    xyz + abc = 1110

end _l508_508738


namespace equation_of_line_AB_lambda_range_l508_508118

-- Definition of the ellipse Γ: (x^2)/4 + (y^2)/2 = 1
def ellipse (P : ℝ × ℝ) : Prop := P.1^2 / 4 + P.2^2 / 2 = 1

-- Conditions for Line l₁, sloping through P(1,1) with a specific slope
def line_through_P (P : ℝ × ℝ) (l : ℝ × ℝ) (k : ℝ) : Prop := l.2 - P.2 = k * (l.1 - P.1)

-- Defining the point P(1,1)
def pointP := (1 : ℝ, 1 : ℝ)

-- Define the property that P(1,1) is the midpoint of segment AB
def midpoint (P A B : ℝ × ℝ) : Prop := P.1 = (A.1 + B.1) / 2 ∧ P.2 = (A.2 + B.2) / 2

-- Problem Statement (1): Find the equation of line AB
theorem equation_of_line_AB 
(A B : ℝ × ℝ) (k : ℝ) 
(h₁ : ellipse A)
(h₂ : ellipse B)
(h₃ : midpoint pointP A B) 
(h₄ : line_through_P pointP A k)
(h₅ : line_through_P pointP B k) :
  (1 : ℝ) + 2 * (B.2 - A.2) / (B.1 - A.1) - 3 = 0 := 
sorry

-- Problem Statement (2): Determine the range of λ = |AB| / |CD|
theorem lambda_range 
(A B C D : ℝ × ℝ) (k : ℝ)
(h₁ : ellipse A)
(h₂ : ellipse B)
(h₃ : ellipse C)
(h₄ : ellipse D)
(h₅ : midpoint pointP A B)
(h₆ : midpoint pointP C D)
(h₇ : line_through_P pointP A k)
(h₈ : line_through_P pointP B k)
(h₉ : line_through_P pointP C (-k))
(h₁₀ : line_through_P pointP D (-k)) :
  sqrt (2 - sqrt 3) ≤ dist A B / dist C D ∧ dist A B / dist C D ≤ sqrt (2 + sqrt 3) :=
sorry

end equation_of_line_AB_lambda_range_l508_508118


namespace number_of_digits_base_ten_l508_508155

theorem number_of_digits_base_ten (x : ℝ) :
  log 3 (log 2 (log 2 (log 2 x))) = 1 → (⌊256 * log 10 (2 : ℝ)⌋ + 1 = 77) :=
by
  intros h
  sorry

end number_of_digits_base_ten_l508_508155


namespace geometric_sequence_S6_l508_508846

def geometric_sum (a : ℕ → ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  if r = 1 then n * a 0
  else a 0 * (1 - r ^ n) / (1 - r)

variables (S : ℕ → ℝ) (a : ℕ → ℝ) (r : ℝ)

-- Condition 1: S_2 = 6
axiom sum_condition_1 : S 2 = 6

-- Condition 2: S_4 = 30
axiom sum_condition_2 : S 4 = 30

-- Prove S_6 = 126
theorem geometric_sequence_S6 : S 6 = 126 :=
sorry

end geometric_sequence_S6_l508_508846


namespace tile_pentomino_iff_m_even_l508_508311

-- Define the necessary conditions.
def is_even (n : ℕ) : Prop := n % 2 = 0
def can_tile_with_pentomino (m : ℕ) : Prop := ∃ P : fin 5 → fin m → Prop, 
  (∀ i j, (P i j → (i < 5 ∧ j < m)) ∧ (P i j ↔ (∃ k : fin 5, (k = i ∨ k = i) ∨ (k = j ∨ k = j)) ) -- Example tiling condition

-- State the Theorem.
theorem tile_pentomino_iff_m_even (m : ℕ) : can_tile_with_pentomino m ↔ is_even m := sorry

end tile_pentomino_iff_m_even_l508_508311


namespace range_t_l508_508475

noncomputable def problem_statement (f : ℝ → ℝ) : Prop :=
  (∀ x ∈ Icc (-2 : ℝ) 2, (f x + f (-x) = 0)) ∧ 
  (∀ x₁ x₂ ∈ Icc (-2 : ℝ) 2, x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) < 0 ) ∧ 
  (∀ t : ℝ, -2 ≤ 1 - t ∧ 1 - t ≤ 2 ∧ -2 ≤ 1 - t^2 ∧ 1 - t^2 ≤ 2 ∧ 
    f (1 - t) + f (1 - t^2) < 0 → (-1 ≤ t ∧ t < 1))

theorem range_t {f : ℝ → ℝ} : problem_statement f :=
sorry

end range_t_l508_508475


namespace average_income_BC_l508_508620

theorem average_income_BC {A_income B_income C_income : ℝ}
  (hAB : (A_income + B_income) / 2 = 4050)
  (hAC : (A_income + C_income) / 2 = 4200)
  (hA : A_income = 3000) :
  (B_income + C_income) / 2 = 5250 :=
by sorry

end average_income_BC_l508_508620


namespace fibonacci_product_expression_l508_508207

-- Defining the Fibonacci sequence.
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => (fib (n + 1)) + (fib n)

-- Lean theorem to express the proof problem
theorem fibonacci_product_expression :
  (∏ k in Finset.range 50, ((fib (k + 1)) / (fib (k + 2)) - (fib (k + 1)) / (fib (k + 3)))) = (fib 50) / (fib 52) := sorry

end fibonacci_product_expression_l508_508207


namespace price_restoration_l508_508639

theorem price_restoration (P : Real) (hP : P > 0) :
  let new_price := 0.85 * P
  let required_increase := ((1 / 0.85) - 1) * 100
  required_increase = 17.65 :=
by 
  sorry

end price_restoration_l508_508639


namespace triangle_inequality_inequality_l508_508582

-- Define a helper function to describe the triangle inequality
def triangle_inequality (a b c : ℝ) : Prop :=
a + b > c ∧ a + c > b ∧ b + c > a

-- Define the main statement
theorem triangle_inequality_inequality (a b c : ℝ) (h_triangle : triangle_inequality a b c):
  a * (b - c) ^ 2 + b * (c - a) ^ 2 + c * (a - b) ^ 2 + 4 * a * b * c > a ^ 3 + b ^ 3 + c ^ 3 :=
sorry

end triangle_inequality_inequality_l508_508582


namespace lins_calculation_l508_508528

theorem lins_calculation (a b : ℤ) (h : a > b) (ha : 0 < a) (hb : 0 < b) :
  1.03 * a - 0.98 * b > a - b := by
    sorry

end lins_calculation_l508_508528


namespace car_miles_per_tankful_in_city_l508_508688

theorem car_miles_per_tankful_in_city :
  ∀ (miles_highway_per_tankful gallons_highway_per_gallon miles_city_per_gallon : ℕ),
  miles_highway_per_tankful = 480 →
  gallons_highway_per_gallon = 20 →
  miles_city_per_gallon = 14 →
  (miles_highway_per_tankful / gallons_highway_per_gallon) * miles_city_per_gallon = 336 :=
by
  assume miles_highway_per_tankful gallons_highway_per_gallon miles_city_per_gallon
  assume H1 : miles_highway_per_tankful = 480
  assume H2 : gallons_highway_per_gallon = 20
  assume H3 : miles_city_per_gallon = 14
  sorry

end car_miles_per_tankful_in_city_l508_508688


namespace pentagon_area_l508_508271

open Function 

/-
Given a convex pentagon FGHIJ with the following properties:
  1. ∠F = ∠G = 100°
  2. JF = FG = GH = 3
  3. HI = IJ = 5
Prove that the area of pentagon FGHIJ is approximately 15.2562 square units.
-/

noncomputable def area_pentagon_FGHIJ : ℝ :=
  let sin100 := Real.sin (100 * Real.pi / 180)
  let area_FGJ := (3 * 3 * sin100) / 2
  let area_HIJ := (5 * 5 * Real.sqrt 3) / 4
  area_FGJ + area_HIJ

theorem pentagon_area : abs (area_pentagon_FGHIJ - 15.2562) < 0.0001 := by
  sorry

end pentagon_area_l508_508271


namespace centroid_distance_l508_508562

-- Define the given conditions and final goal
theorem centroid_distance (a b c p q r : ℝ) 
  (ha : a ≠ 0)  (hb : b ≠ 0)  (hc : c ≠ 0)
  (centroid : p = a / 3 ∧ q = b / 3 ∧ r = c / 3) 
  (plane_distance : (1 / (1 / a^2 + 1 / b^2 + 1 / c^2).sqrt) = 2) :
  (1 / p^2 + 1 / q^2 + 1 / r^2) = 2.25 := 
by 
  -- Start proof here
  sorry

end centroid_distance_l508_508562


namespace tan_alpha_plus_pi_over_6_l508_508474

theorem tan_alpha_plus_pi_over_6 (α : ℝ) (h1 : 0 < α) (h2 : α < π / 2)
  (h3 : sqrt 3 * sin α + cos α = 8 / 5) : 
  tan (α + π / 6) = 4 / 3 := 
sorry

end tan_alpha_plus_pi_over_6_l508_508474


namespace distance_traveled_l508_508331

-- Given conditions
def speed : ℕ := 100 -- Speed in km/hr
def time : ℕ := 5    -- Time in hours

-- The goal is to prove the distance traveled is 500 km
theorem distance_traveled : speed * time = 500 := by
  -- we state the proof goal
  sorry

end distance_traveled_l508_508331


namespace decreasing_function_l508_508482

noncomputable def f (a x : ℝ) : ℝ := a^(1 - x)

theorem decreasing_function (a : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1) (h₃ : ∀ x > 1, f a x < 1) :
  ∀ x y : ℝ, x < y → f a x > f a y :=
sorry

end decreasing_function_l508_508482


namespace find_integer_pairs_l508_508078

theorem find_integer_pairs (x y: ℤ) :
  x^2 - y^4 = 2009 → (x = 45 ∧ (y = 2 ∨ y = -2)) ∨ (x = -45 ∧ (y = 2 ∨ y = -2)) :=
by
  sorry

end find_integer_pairs_l508_508078


namespace correct_definition_of_skew_lines_l508_508047

def skew_lines (l1 l2 : Line) : Prop :=
  ¬ (∃ p : Point, p ∈ l1 ∧ p ∈ l2) ∧ ¬ parallel l1 l2

def option_A (l1 : Line) (P : Plane) : Prop :=
  l1 ∈ P ∧ (∃ l2 : Line, ¬ (l2 ∈ P))

def option_B (l1 l2 : Line) : Prop :=
  ∃ (P Q : Plane), l1 ∈ P ∧ l2 ∈ Q ∧ ¬ (P = Q)

def option_C (l1 l2 : Line) : Prop :=
  ¬ (∃ P : Plane, l1 ∈ P ∧ l2 ∈ P)

def option_D (l1 l2 : Line) : Prop :=
  ¬ (∃ P : Plane, l1 ∈ P ∧ l2 ∈ P) ∧ skew_lines l1 l2

theorem correct_definition_of_skew_lines (l1 l2 : Line) :
  skew_lines l1 l2 → option_D l1 l2 :=
by
  sorry

end correct_definition_of_skew_lines_l508_508047


namespace auditorium_sampling_method_l508_508534

/-- In a small auditorium, there are 25 rows and each row has 20 seats. The auditorium was filled with students.
After the lecture, 25 students sitting on seat number 15 were selected for a test. This method of selection
is the "systematic sampling method". -/
theorem auditorium_sampling_method (rows seats : ℕ) [h_rows : rows = 25] [h_seats : seats = 20] 
  (filled : ∀ (r : ℕ), r < rows → ∀ (s : ℕ), s < seats → ∃ student : ℕ, student) :
  ∃ method : String, method = "systematic sampling method" :=
by
  obtain ⟨seats_each_row, _⟩ :=  h_seats,
  sorry

end auditorium_sampling_method_l508_508534


namespace inversely_proportional_value_l508_508261

theorem inversely_proportional_value (a b k : ℝ) (h1 : a * b = k) (h2 : a = 40) (h3 : b = 8) :
  ∃ a' : ℝ, a' * 10 = k ∧ a' = 32 :=
by {
  use 32,
  sorry
}

end inversely_proportional_value_l508_508261


namespace volume_conversion_l508_508015

theorem volume_conversion :
  (∀ vol_ft³ : ℝ, vol_ft³ = 256 → vol_ft³ * 0.0283 ≈ 7.245) :=
by
  sorry

end volume_conversion_l508_508015


namespace correct_statement_l508_508327

section
variables {a b c d : Real}

-- Define the conditions as hypotheses/functions

-- Statement A: If a > b, then 1/a < 1/b
def statement_A (a b : Real) : Prop := a > b → 1 / a < 1 / b

-- Statement B: If a > b, then a^2 > b^2
def statement_B (a b : Real) : Prop := a > b → a^2 > b^2

-- Statement C: If a > b and c > d, then ac > bd
def statement_C (a b c d : Real) : Prop := a > b ∧ c > d → a * c > b * d

-- Statement D: If a^3 > b^3, then a > b
def statement_D (a b : Real) : Prop := a^3 > b^3 → a > b

-- The Lean statement to prove which statement is correct
theorem correct_statement : ¬ statement_A a b ∧ ¬ statement_B a b ∧ ¬ statement_C a b c d ∧ statement_D a b :=
by {
  sorry
}

end

end correct_statement_l508_508327


namespace max_FM_l508_508897

noncomputable def maxF (M : ℕ) : ℕ :=
  let a := M / 1000
  let b := (M % 1000) / 100
  let c := (M % 100) / 10
  let d := M % 10
  if b + c + d = 12 ∧ a = b - d ∧ (F(M) / 9).isInt then
    9
  else
    sorry -- other cases

-- Definitions of M, N, F
def F (M : ℕ) : ℤ :=
  let a := M / 1000
  let b := (M % 1000) / 100
  let c := (M % 100) / 10
  let d := M % 10
  let N := 1000 * b + 100 * a + 10 * d + c
  (M - N : ℤ) / 9

-- The problem statement in Lean 4
theorem max_FM (M : ℕ) (h1 : ∃ a b c d : ℕ, M = 1000 * a + 100 * b + 10 * c + d ∧ b + c + d = 12 ∧ a = b - d)
  (h2 : (F M / 9).isInt) : maxF M = 9 := by
  sorry

end max_FM_l508_508897


namespace sum_of_base4_numbers_l508_508428

theorem sum_of_base4_numbers :
  let n1 := (1 * 4^2 + 3 * 4^1 + 2 * 4^0)
  let n2 := (2 * 4^2 + 0 * 4^1 + 3 * 4^0)
  let n3 := (3 * 4^2 + 2 * 4^1 + 1 * 4^0)
  let n4 := (1 * 4^2 + 2 * 4^1 + 0 * 4^0)
  let sum := n1 + n2 + n3 + n4
  let result := (2 * 4^3 + 0 * 4^2 + 1 * 4^1 + 0 * 4^0)
  (sum = result) :=
by
  let n1 := (1 * 4^2 + 3 * 4^1 + 2 * 4^0)
  let n2 := (2 * 4^2 + 0 * 4^1 + 3 * 4^0)
  let n3 := (3 * 4^2 + 2 * 4^1 + 1 * 4^0)
  let n4 := (1 * 4^2 + 2 * 4^1 + 0 * 4^0)
  let sum := n1 + n2 + n3 + n4
  let result := (2 * 4^3 + 0 * 4^2 + 1 * 4^1 + 0 * 4^0)
  show sum = result from sorry

end sum_of_base4_numbers_l508_508428


namespace max_numbers_greater_than_15_l508_508268

-- Let n be the number of elements and avg be the average of these elements.
def max_numbers_greater_than (n : ℕ) (avg : ℕ) (threshold : ℕ) : Prop :=
  (\u_total A₀ > n/2)
  (sum_of_list A₀ = n * avg) : 
  (\u_A₀ i > A₀) 

theorem max_numbers_greater_than_15 :
  max_numbers_greater_than 30 4 4 15 := by
  sorry

end max_numbers_greater_than_15_l508_508268


namespace cost_price_computer_table_l508_508337

variable (CP SP : ℝ)

theorem cost_price_computer_table (h1 : SP = 2 * CP) (h2 : SP = 1000) : CP = 500 := by
  sorry

end cost_price_computer_table_l508_508337


namespace lines_concurrent_l508_508203

variables {α : Type*} [IncirculatedQuadrilateral α]

namespace AB_CD_AD_BC_lines_concurrent

noncomputable def are_lines_concurrent (ABCD : Quadrilateral) 
(X1 Y1 X2 Y2 X3 Y3 : α) : Prop :=
  X1 ∈ circle_diameter ABCD.A ABCD.B ∧ Y1 ∈ circle_diameter ABCD.C ABCD.D ∧
  X1 ∈ circle_diameter ABCD.A ABCD.B ∧ Y1 ∈ circle_diameter ABCD.C ABCD.D ∧
  X2 ∈ circle_diameter ABCD.B ABCD.C ∧ Y2 ∈ circle_diameter ABCD.A ABCD.D ∧
  X3 ∈ circle_diameter ABCD.A ABCD.C ∧ Y3 ∈ circle_diameter ABCD.B ABCD.D ∧
  concurrent (line_through X1 Y1) (line_through X2 Y2) (line_through X3 Y3)

theorem lines_concurrent (ABCD : Quadrilateral) (X1 Y1 X2 Y2 X3 Y3 : α) 
(hX1 : X1 ∈ circle_diameter ABCD.A ABCD.B) (hY1 : Y1 ∈ circle_diameter ABCD.C ABCD.D)
(hX2 : X2 ∈ circle_diameter ABCD.B ABCD.C) (hY2 : Y2 ∈ circle_diameter ABCD.A ABCD.D)
(hX3 : X3 ∈ circle_diameter ABCD.A ABCD.C) (hY3 : Y3 ∈ circle_diameter ABCD.B ABCD.D) :
  concurrent (line_through X1 Y1) (line_through X2 Y2) (line_through X3 Y3) :=
by
  sorry

end lines_concurrent_l508_508203


namespace lees_friend_initial_money_l508_508200

theorem lees_friend_initial_money (lee_initial_money friend_initial_money total_cost change : ℕ) 
  (h1 : lee_initial_money = 10) 
  (h2 : total_cost = 15) 
  (h3 : change = 3) 
  (h4 : (lee_initial_money + friend_initial_money) - total_cost = change) : 
  friend_initial_money = 8 := by
  sorry

end lees_friend_initial_money_l508_508200


namespace blake_change_given_l508_508750

theorem blake_change_given :
  let oranges := 40
  let apples := 50
  let mangoes := 60
  let total_amount := 300
  let total_spent := oranges + apples + mangoes
  let change_given := total_amount - total_spent
  change_given = 150 :=
by
  sorry

end blake_change_given_l508_508750


namespace marathon_yards_l508_508701

theorem marathon_yards (miles_per_marathon : ℕ) (yards_per_marathon : ℕ) (yards_per_mile : ℕ) (marathons_run : ℕ) 
  (total_miles : ℕ) (total_yards : ℕ) (h1 : miles_per_marathon = 26) (h2 : yards_per_marathon = 385)
  (h3 : yards_per_mile = 1760) (h4 : marathons_run = 15) (h5 : 
  total_miles = marathons_run * miles_per_marathon + (marathons_run * yards_per_marathon) / yards_per_mile) 
  (h6 : total_yards = (marathons_run * yards_per_marathon) % yards_per_mile) : 
  total_yards = 495 :=
by
  -- This will be our process to verify the transformation
  sorry

end marathon_yards_l508_508701


namespace line_equation_l508_508625

theorem line_equation (x y : ℝ) (h : ∀ x : ℝ, (x - 2) * 1 = y) : x - y - 2 = 0 :=
sorry

end line_equation_l508_508625


namespace count_solutions_inequalities_l508_508503

theorem count_solutions_inequalities :
  {x : ℤ | -5 * x ≥ 2 * x + 10} ∩ {x : ℤ | -3 * x ≤ 15} ∩ {x : ℤ | -6 * x ≥ 3 * x + 21} = {x : ℤ | x = -5 ∨ x = -4 ∨ x = -3} :=
by 
  sorry

end count_solutions_inequalities_l508_508503


namespace product_floor_ceil_sequence_l508_508410

noncomputable def floor (x : ℝ) : ℤ := Int.floor x
noncomputable def ceil (x : ℝ) : ℤ := Int.ceil x

theorem product_floor_ceil_sequence :
    (floor (-6 - 0.5) * ceil (6 + 0.5)) *
    (floor (-5 - 0.5) * ceil (5 + 0.5)) *
    (floor (-4 - 0.5) * ceil (4 + 0.5)) *
    (floor (-3 - 0.5) * ceil (3 + 0.5)) *
    (floor (-2 - 0.5) * ceil (2 + 0.5)) *
    (floor (-1 - 0.5) * ceil (1 + 0.5)) *
    (floor (-0.5) * ceil (0.5)) = -25401600 :=
by
  sorry

end product_floor_ceil_sequence_l508_508410


namespace ratio_of_a_b_l508_508585

theorem ratio_of_a_b (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : ∃ c : ℝ, (3 - 5 * complex.i) * (a + b * complex.i) = c * complex.i) : a / b = -5 / 3 :=
sorry

end ratio_of_a_b_l508_508585


namespace log_function_domain_l508_508623

theorem log_function_domain {x : ℝ} (h : 1 / x - 1 > 0) : 0 < x ∧ x < 1 :=
sorry

end log_function_domain_l508_508623


namespace discount_difference_l508_508020

def original_price (P : ℝ) : ℝ := P
def first_discounted_price (P : ℝ) : ℝ := 0.75 * P
def second_discounted_price (P : ℝ) : ℝ := 0.85 * (first_discounted_price P)
def final_discounted_price (P : ℝ) : ℝ := 0.90 * (second_discounted_price P)
def true_discount (P : ℝ) : ℝ := P - (final_discounted_price P)
def true_discount_percentage (P : ℝ) : ℝ := (true_discount P / P) * 100

theorem discount_difference (P : ℝ) (hP : 0 < P) : 
  abs (45 - true_discount_percentage P) = 2.375 :=
by { sorry }

end discount_difference_l508_508020


namespace danny_bottle_cap_count_l508_508050

theorem danny_bottle_cap_count 
  (initial_caps : Int) 
  (found_caps : Int) 
  (final_caps : Int) 
  (h1 : initial_caps = 6) 
  (h2 : found_caps = 22) 
  (h3 : final_caps = initial_caps + found_caps) : 
  final_caps = 28 :=
by
  sorry

end danny_bottle_cap_count_l508_508050


namespace total_annual_salary_excluding_turban_l508_508499

-- Let X be the total amount of money Gopi gives as salary for one year, excluding the turban.
variable (X : ℝ)

-- Condition: The servant leaves after 9 months and receives Rs. 60 plus the turban.
variable (received_money : ℝ)
variable (turban_price : ℝ)

-- Condition values:
axiom received_money_condition : received_money = 60
axiom turban_price_condition : turban_price = 30

-- Question: Prove that X equals 90.
theorem total_annual_salary_excluding_turban :
  3/4 * (X + turban_price) = 90 :=
sorry

end total_annual_salary_excluding_turban_l508_508499


namespace no_such_functions_exist_l508_508779

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

theorem no_such_functions_exist : ¬ (∃ f g : ℝ → ℝ, ∀ x y : ℝ, f(x + f(y)) = y^2 + g(x)) :=
by
  assume h
  obtain ⟨f, g, h⟩ := h
  sorry

end no_such_functions_exist_l508_508779


namespace pencil_cost_3000_l508_508686

theorem pencil_cost_3000
  (cost_per_box : ℝ)
  (box_size : ℕ)
  (total_pencils : ℕ)
  (discount_rate : ℝ) :
  cost_per_box = 30 ∧ box_size = 100 ∧ total_pencils = 3000 ∧ discount_rate = 0.1 →
  let cost_per_pencil := cost_per_box / box_size in
  let total_cost_without_discount := cost_per_pencil * total_pencils in
  let discount := discount_rate * total_cost_without_discount in
  let total_cost_after_discount := total_cost_without_discount - discount in
  total_cost_after_discount = 810 := 
sorry

end pencil_cost_3000_l508_508686


namespace correct_statement_is_D_l508_508725

-- Define the statements
def statementA : Prop := ∀ (data : List ℝ) (histogram_area_eq_frequency : bool), histogram_area_eq_frequency = false
def statementB : Prop := ∀ (data : List ℝ) (variance : ℝ), stddev data = variance^2
def statementC : Prop := let data1 := [2, 3, 4, 5]; let data2 := [4, 6, 8, 10] in variance data1 = variance data2 / 2
def statementD : Prop := ∀ (data : List ℝ), higher_variance_implies_greater_fluctuation data

-- Assume some context where these variables and relations hold, for illustration:
constant stddev : List ℝ → ℝ
constant variance : List ℝ → ℝ
constant higher_variance_implies_greater_fluctuation : List ℝ → Prop

-- Main theorem to prove
theorem correct_statement_is_D : statementD :=
sorry

end correct_statement_is_D_l508_508725


namespace radish_carrot_ratio_l508_508646

theorem radish_carrot_ratio (num_cucumbers : ℕ) (num_carrots : ℕ) 
  (h1 : num_cucumbers = 15) (h2 : num_carrots = 9) 
  (h3 : ∃ num_radishes, num_radishes = 3 * num_cucumbers) :
  ∃ k, k = num_radishes / num_carrots ∧ k = 5 :=
by
  rcases h3 with ⟨num_radishes, hr⟩
  use 5
  have h4 : num_radishes = 45 := by rw [hr, h1]; norm_num
  have h5 : num_radishes / num_carrots = 5 := by rw [h4, h2]; norm_num
  exact ⟨h5, by norm_num⟩

end radish_carrot_ratio_l508_508646


namespace JakePresentWeight_l508_508159

def JakeWeight (J S : ℕ) : Prop :=
  J - 33 = 2 * S ∧ J + S = 153

theorem JakePresentWeight : ∃ (J : ℕ), ∃ (S : ℕ), JakeWeight J S ∧ J = 113 := 
by
  sorry

end JakePresentWeight_l508_508159


namespace allowable_rectangular_formations_count_l508_508361

theorem allowable_rectangular_formations_count (s t f : ℕ) 
  (h1 : s * t = 240)
  (h2 : Nat.Prime s)
  (h3 : 8 ≤ t ∧ t ≤ 30)
  (h4 : f ≤ 8)
  : f = 0 :=
sorry

end allowable_rectangular_formations_count_l508_508361


namespace find_B_sin_squared_sum_range_l508_508809

-- Define the angles and vectors
variables {A B C : ℝ}
variables (m n : ℝ × ℝ)
variables (α : ℝ)

-- Basic triangle angle sum condition
axiom angle_sum : A + B + C = Real.pi

-- Define vectors as per the problem statement
axiom vector_m : m = (Real.sin B, 1 - Real.cos B)
axiom vector_n : n = (2, 0)

-- The angle between vectors m and n is π/3
axiom angle_between_vectors : α = Real.pi / 3
axiom angle_condition : Real.cos α = (2 * Real.sin B + 0 * (1 - Real.cos B)) / 
                                     (Real.sqrt (Real.sin B ^ 2 + (1 - Real.cos B) ^ 2) * 2)

theorem find_B : B = 2 * Real.pi / 3 := 
sorry

-- Conditions for range of sin^2 A + sin^2 C
axiom range_condition : (0 < A ∧ A < Real.pi / 3) 
                     ∧ (0 < C ∧ C < Real.pi / 3)
                     ∧ (A + C = Real.pi / 3)

theorem sin_squared_sum_range : (Real.sin A) ^ 2 + (Real.sin C) ^ 2 ∈ Set.Ico (1 / 2) 1 := 
sorry

end find_B_sin_squared_sum_range_l508_508809


namespace sum_of_first_twelve_terms_l508_508429

section ArithmeticSequence

variables (a : ℕ → ℚ) (d : ℚ) (a₁ : ℚ)

-- General definition of the nth term in arithmetic progression
def arithmetic_term (n : ℕ) : ℚ := a₁ + (n - 1) * d

-- Given conditions in the problem
axiom fifth_term : arithmetic_term a₁ d 5 = 1
axiom seventeenth_term : arithmetic_term a₁ d 17 = 18

-- Define the sum of the first n terms in arithmetic sequence
def sum_arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  n * (2 * a₁ + (n - 1) * d) / 2

-- Statement of the proof problem
theorem sum_of_first_twelve_terms : 
  sum_arithmetic_sequence a₁ d 12 = 37.5 := 
sorry

end ArithmeticSequence

end sum_of_first_twelve_terms_l508_508429


namespace scatter_plot_positive_correlation_l508_508523

/-- 
Given that the scatter plot of two variables goes from the bottom left corner to the top right corner,
prove that these two variables are positively correlated.
-/
theorem scatter_plot_positive_correlation (x y : ℝ → ℝ) 
  (h : ∀ t, x t ≤ y t) : 
  positively_correlated x y :=
sorry

end scatter_plot_positive_correlation_l508_508523


namespace chord_length_of_arc_and_angle_l508_508459

noncomputable def chord_length (m : Real) (θ : Real) : Real :=
  let r := m * 180 / (θ * Real.pi),
  2 * r * (Real.sin (θ / 2 * Real.pi / 180))

theorem chord_length_of_arc_and_angle (m : Real) : chord_length m 120 = (3 * Real.sqrt 3 / (4 * Real.pi)) * m :=
by
  -- Proof goes here
  sorry

end chord_length_of_arc_and_angle_l508_508459


namespace stratified_sampling_second_year_l508_508183

-- Define the numbers of students in the first year and the second year
def num_students_first_year : ℕ := 30
def num_students_second_year : ℕ := 40

-- Define the number of students selected from the first year
def selected_students_first_year : ℕ := 6

-- Use the stratified sampling ratio
def selection_ratio : ℚ := selected_students_first_year / num_students_first_year

-- Define the expected number of students to be selected from the second year
def expected_selected_students_second_year : ℕ := 40 * selection_ratio.toNat

-- Prove that the number of students selected from the second year equals 8
theorem stratified_sampling_second_year :
  expected_selected_students_second_year = 8 :=
by
  sorry

end stratified_sampling_second_year_l508_508183


namespace cara_pairs_between_l508_508044

theorem cara_pairs_between (n : ℕ) (h : n = 5) : nat.choose n 2 = 10 :=
by {
  rw h,
  exact nat.choose_succ_succ 4 1,
  sorry
}

end cara_pairs_between_l508_508044


namespace anna_salads_l508_508731

def num_plants : ℕ := 8
def salads_per_plant : ℕ := 3
def loss_fraction : ℚ := 1 / 2

theorem anna_salads : 
  let total_salads := num_plants * salads_per_plant in
  let remaining_salads := (total_salads : ℚ) * (1 - loss_fraction) in
  remaining_salads = 12 := 
by
  sorry

end anna_salads_l508_508731


namespace sum_of_first_twelve_terms_l508_508430

section ArithmeticSequence

variables (a : ℕ → ℚ) (d : ℚ) (a₁ : ℚ)

-- General definition of the nth term in arithmetic progression
def arithmetic_term (n : ℕ) : ℚ := a₁ + (n - 1) * d

-- Given conditions in the problem
axiom fifth_term : arithmetic_term a₁ d 5 = 1
axiom seventeenth_term : arithmetic_term a₁ d 17 = 18

-- Define the sum of the first n terms in arithmetic sequence
def sum_arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  n * (2 * a₁ + (n - 1) * d) / 2

-- Statement of the proof problem
theorem sum_of_first_twelve_terms : 
  sum_arithmetic_sequence a₁ d 12 = 37.5 := 
sorry

end ArithmeticSequence

end sum_of_first_twelve_terms_l508_508430


namespace ways_to_sum_91_l508_508881

theorem ways_to_sum_91 :
  (∃ S : set (finset ℕ), (∀ s ∈ S, (finset.sum s id = 91) ∧ (s.card ≥ 2)) ∧ S.card = 3) :=
by
  sorry

end ways_to_sum_91_l508_508881


namespace henry_time_around_track_l508_508144

theorem henry_time_around_track (H : ℕ) : 
  (∀ (M := 12), lcm M H = 84) → H = 7 :=
by
  sorry

end henry_time_around_track_l508_508144


namespace initial_percentage_of_alcohol_l508_508685

theorem initial_percentage_of_alcohol (P : ℚ) :
  let initial_volume := 6
  let added_alcohol := 3.6
  let final_volume := initial_volume + added_alcohol
  let final_percentage := 0.5
  let final_alcohol := final_volume * final_percentage
  initial_volume * (P / 100) + added_alcohol = final_alcohol
  → P = 20 :=
by
  intro h
  have h1 : initial_volume = 6 := rfl
  have h2 : added_alcohol = 3.6 := rfl
  have h3 : final_volume = initial_volume + added_alcohol := rfl
  have h4 : final_percentage = 0.5 := rfl
  have h5 : final_alcohol = final_volume * final_percentage := rfl
  have eq1 : initial_volume * (P / 100) + added_alcohol = final_volume * final_percentage := by assumption
  sorry

end initial_percentage_of_alcohol_l508_508685


namespace average_weight_increase_l508_508181

theorem average_weight_increase 
  (n : ℕ) (old_weight new_weight : ℝ) (group_size := 8) 
  (old_weight := 70) (new_weight := 90) : 
  ((new_weight - old_weight) / group_size) = 2.5 := 
by sorry

end average_weight_increase_l508_508181


namespace sum_f_values_l508_508167

noncomputable def f (ω x : ℝ) := sin (ω * x) + (sqrt 3) * cos (ω * x)

theorem sum_f_values :
  ∃ ω > 0, (∀ x y, f ω x = 0 → f ω y = 0 → abs (x - y) = 2) → (f ω 1 + f ω 2 + f ω 3 + f ω 4 + f ω 5 + f ω 6 + f ω 7 + f ω 8 + f ω 9 = 1) := 
begin 
  sorry
end

end sum_f_values_l508_508167


namespace minimum_chord_length_intercepted_by_circle_and_line_l508_508096

noncomputable def minimum_chord_length := 2 * Real.sqrt 6

theorem minimum_chord_length_intercepted_by_circle_and_line (a b : ℝ) (ha : ¬ (a = 0 ∧ b = 0)) :
  ∃ l C,
  (C : ∀ x y : ℝ, x^2 + y^2 = 16) ∧ (l : ∀ x y : ℝ, (a - b) * x + (3 * b - 2 * a) * y - a = 0) ∧
  (∀ x y : ℝ, l x y → C x y) →
  (∀ (p1 p2 : ℝ × ℝ), p1 ≠ p2 → (l p1.fst p1.snd → l p2.fst p2.snd → Real.dist p1 p2 ≥ minimum_chord_length)) :=
sorry

end minimum_chord_length_intercepted_by_circle_and_line_l508_508096


namespace pipeline_equation_correct_l508_508178

variables (m x n : ℝ) -- Length of the pipeline, kilometers per day, efficiency increase percentage
variable (h : 0 < n) -- Efficiency increase percentage is positive

theorem pipeline_equation_correct :
  (m / x) - (m / ((1 + (n / 100)) * x)) = 8 :=
sorry -- Proof omitted

end pipeline_equation_correct_l508_508178


namespace euler_lines_concurrent_l508_508202

-- Definition of the Euler line of a triangle
def euler_line (T : Triangle) : Line := sorry

-- Fermat point definition in the context of a triangle
def fermat_point (T : Triangle) : Point := sorry

-- Given conditions
variables {A B C : Point}
variables (ABC : Triangle A B C)
variables (all_angles_le_120 : ∀ α ∈ angles ABC, α ≤ 120)
variables (F : Point)
variables (F_fermat_point : fermat_point ABC = F)
variables (angle_AFB_eq_120 : angle A F B = 120)
variables (angle_BFC_eq_120 : angle B F C = 120)
variables (angle_CFA_eq_120 : angle C F A = 120)

-- The theorem to prove
theorem euler_lines_concurrent :
  are_concurrent (euler_line (Triangle.mk B F C)) (euler_line (Triangle.mk C F A)) (euler_line (Triangle.mk A F B)) :=
sorry

end euler_lines_concurrent_l508_508202


namespace problem_statement_l508_508531

noncomputable def area_of_circle (O AB CD DF : Type) [MetricSpace O] 
  [IsCircle O] (A B C D E F : O) [IsDiameter AB] [IsDiameter CD] 
  [IsPerpendicular AB CD] [IsChord DF] (h_intersect : Intersects DF AB E) 
  (h_DE : Distance D E = 8) (h_EF : Distance E F = 4) : ℝ :=
by
  sorry

theorem problem_statement {O : Type} [MetricSpace O] 
  [IsCircle O] (A B C D E F : O) [IsDiameter AB] [IsDiameter CD] 
  [IsPerpendicular AB CD] [IsChord DF] (h_intersect : Intersects DF AB E) 
  (h_DE : Distance D E = 8) (h_EF : Distance E F = 4) :
  area_of_circle O AB CD DF A B C D E F h_intersect h_DE h_EF = 32 * Real.pi :=
by
  sorry

end problem_statement_l508_508531


namespace distance_AF_l508_508466

-- define the focus of the parabola y^2 = 4x
structure Point :=
  (x : ℝ)
  (y : ℝ)

def focus : Point := ⟨1, 0⟩

def on_parabola (A : Point) : Prop :=
  A.y ^ 2 = 4 * A.x

def midpoint_abscissa (A : Point) (F : Point) : ℝ :=
  (A.x + F.x) / 2

theorem distance_AF
  (A : Point)
  (hA : on_parabola A)
  (h_midpoint : midpoint_abscissa A focus = 2) :
  dist (A.x, A.y) (focus.x, focus.y) = 4 :=
sorry

end distance_AF_l508_508466


namespace integer_values_in_interval_l508_508502

theorem integer_values_in_interval : (∃ n : ℕ, n = 25 ∧ ∀ x : ℤ, abs x < 4 * π ↔ -12 ≤ x ∧ x ≤ 12) :=
by
  sorry

end integer_values_in_interval_l508_508502


namespace distinct_triangles_2x4_grid_l508_508147

theorem distinct_triangles_2x4_grid : 
  let total_points := 8 in
  let rows := 2 in
  let cols := 4 in
  let total_combinations := Nat.choose total_points 3 in
  let degenerate_cases :=
    let row_degenerate := rows * Nat.choose cols 3 in
    let col_degenerate := cols  in
    row_degenerate + col_degenerate in
  total_combinations - degenerate_cases = 44 :=
by
  sorry

end distinct_triangles_2x4_grid_l508_508147


namespace distinct_valid_c_values_l508_508803

theorem distinct_valid_c_values : 
  let is_solution (c : ℤ) (x : ℚ) := (5 * ⌊x⌋₊ + 3 * ⌈x⌉₊ = c) 
  ∃ s : Finset ℤ, (∀ c ∈ s, (∃ x : ℚ, is_solution c x)) ∧ s.card = 500 :=
by sorry

end distinct_valid_c_values_l508_508803


namespace limit_fraction_power_eq_half_l508_508041

open Real

theorem limit_fraction_power_eq_half :
  Tendsto (fun x => ( (1 + 8 * x) / (2 + 11 * x) ) ^ (1 / (x^2 + 1))) (nhds 0) (nhds (1 / 2)) :=
  sorry

end limit_fraction_power_eq_half_l508_508041


namespace puppies_in_each_cage_l508_508010

theorem puppies_in_each_cage (initial_puppies sold_puppies cages : ℕ)
  (h_initial : initial_puppies = 18)
  (h_sold : sold_puppies = 3)
  (h_cages : cages = 3) :
  (initial_puppies - sold_puppies) / cages = 5 :=
by
  sorry

end puppies_in_each_cage_l508_508010


namespace find_DC_l508_508974

variables (C D : Matrix (Fin 2) (Fin 2) ℝ)
def matrix_C_plus_D_eq_CD : Prop := C + D = C ⬝ D
def matrix_CD_value : Prop := C ⬝ D = ![![10, 2], ![-4, 4]]

theorem find_DC (h1 : matrix_C_plus_D_eq_CD C D) (h2 : matrix_CD_value C D) : 
  D ⬝ C = ![![10, 2], ![-4, 4]] :=
sorry

end find_DC_l508_508974


namespace g_at_1_l508_508510

variable (g : ℝ → ℝ)

theorem g_at_1 (h : ∀ x : ℝ, g (2 * x - 5) = 3 * x + 9) : g 1 = 18 := by
  sorry

end g_at_1_l508_508510


namespace circle_area_ratio_in_hexagon_l508_508764

/-- Math problem statement based on regular hexagon and circle tangency properties. -/
theorem circle_area_ratio_in_hexagon :
  ∃ (hexagon_side_length : ℝ) (r1 r2 : ℝ),
    hexagon_side_length = 2 ∧
    r1 = (Real.sqrt 3) / 3 ∧ 
    r2 = (Real.sqrt 3) ∧ 
    let A1 := π * r1^2 
    let A2 := π * r2^2 
    let ratio := A2 / A1 
    ratio = 3 * Real.sqrt 3 :=
sorry

end circle_area_ratio_in_hexagon_l508_508764


namespace eccentricity_of_ellipse_l508_508469

-- Define the parameters in the conditions
variables {a b : ℝ} (h : a > b > 0)
variables {F1 F2 P : ℝ} -- point on ellipse and foci
variables (h1 : |P - F1| - |P - F2| = 2 * b)
variables (h2 : |P - F1| * |P - F2| = 3 / 2 * a * b)

-- Define the ellipses
def ellipse (a b : ℝ) := ∃ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1)

-- Actual proof problem statement
theorem eccentricity_of_ellipse : ellipse a b → ∃ e : ℝ, e = real.sqrt 3 / 2 :=
by 
  sorry

end eccentricity_of_ellipse_l508_508469


namespace period_of_f_min_value_of_f_l508_508852

-- Define the given function.
def f (x : ℝ) : ℝ := sqrt 2 * sin (x / 2) * cos (x / 2) - sqrt 2 * (sin (x / 2))^2

-- Prove the smallest positive period of f(x) is 2π.
theorem period_of_f : ∀ x : ℝ, f (x + 2 * π) = f x := by
  sorry

-- Prove the minimum value of f(x) in the interval [-π, 0] is -1 - (sqrt 2) / 2.
theorem min_value_of_f : ∀ x : ℝ, -π ≤ x ∧ x ≤ 0 → f x ≥ f (-3 * π / 4) ∧ (f (-3 * π / 4) = -1 - sqrt 2 / 2) := by
  sorry

end period_of_f_min_value_of_f_l508_508852


namespace tank_capacity_l508_508359

-- Define the conditions
def leak_rate (C : ℝ) : ℝ := C / 6
def inlet_rate : ℝ := 4.5 * 60  -- 270 liters per hour
def net_emptying_rate (C : ℝ) : ℝ := C / 8

-- Define the target proof problem
theorem tank_capacity : ∃ C : ℝ, leak_rate C + net_emptying_rate C = inlet_rate ∧ C ≈ 925.71 :=
by
  sorry

end tank_capacity_l508_508359


namespace uniquely_determined_a_square_plus_2a_l508_508586

theorem uniquely_determined_a_square_plus_2a (a b t : ℝ) (h1 : |a + 1| = t) (h2 : |Real.sin b| = t) :
  (∃ t, ∀ p : ℝ, p = t^2 - 1 → p = a^2 + 2a) :=
by
  sorry

end uniquely_determined_a_square_plus_2a_l508_508586


namespace tangent_line_eq_root_distance_bound_l508_508855

noncomputable
def f (x : ℝ) : ℝ := 3 * (1 - x) * Real.log (1 + x) + Real.sin (Real.pi * x)

theorem tangent_line_eq (f : ℝ → ℝ) (H : ∀ x, f x = 3 * (1 - x) * Real.log (1 + x) + Real.sin (Real.pi * x)) :
  let y := f 0
  let m := deriv f 0
  m = Real.pi + 3 → y = 0 → ∀ x, y = (Real.pi + 3) * x := sorry

theorem root_distance_bound (m : ℝ) (h : ∀ x, f x = 3 * (1 - x) * Real.log (1 + x) + Real.sin (Real.pi * x)) 
  (h1 : 0 ≤ x1) (h2 : x1 < 1) (h3 : 0 ≤ x2) (h4 : x2 ≤ 1) (h5 : x1 ≠ x2) :
  f x1 = m → f x2 = m → |x1 - x2| ≤ 1 - (2 * m) / (Real.pi + 3) := sorry

end tangent_line_eq_root_distance_bound_l508_508855


namespace carol_rectangle_length_l508_508759

theorem carol_rectangle_length (lCarol : ℝ) :
    (∃ (wCarol : ℝ), wCarol = 20 ∧ lCarol * wCarol = 300) ↔ lCarol = 15 :=
by
  have jordan_area : 6 * 50 = 300 := by norm_num
  sorry

end carol_rectangle_length_l508_508759


namespace set_intersection_union_eq_complement_l508_508866

def A : Set ℝ := {x | 2 * x^2 + x - 3 = 0}
def B : Set ℝ := {i | i^2 ≥ 4}
def complement_C : Set ℝ := {-1, 1, 3/2}

theorem set_intersection_union_eq_complement :
  A ∩ B ∪ complement_C = complement_C :=
by
  sorry

end set_intersection_union_eq_complement_l508_508866


namespace sum_y_sequence_eq_l508_508770

noncomputable def y_sequence (m : ℕ) : ℕ → ℝ
| 0     := 1
| 1     := m
| (k+2) := ((m + 1) * y_sequence m (k + 1) + (m - k) * y_sequence m k) / (k + 2)

theorem sum_y_sequence_eq (m : ℕ) : 
  ∑ (k : ℕ) in Finset.range (m + 1), y_sequence m k = 2^(m+1) := by sorry

end sum_y_sequence_eq_l508_508770


namespace length_AF_is_4_l508_508468

-- Define the focus and the parabola y^2 = 4x
def parabola (y x : ℝ) : Prop := y^2 = 4 * x

-- Define the focus of the parabola
def is_focus (F : ℝ × ℝ) : Prop := F = (1, 0)

-- Define a point on the parabola
def is_point_on_parabola (A : ℝ × ℝ) : Prop := ∃ y, parabola y (A.fst)

-- Define the midpoint condition
def midpoint_condition (A F : ℝ × ℝ) : Prop := (A.fst + F.fst) / 2 = 2

-- Define the distance function
noncomputable def distance (A F : ℝ × ℝ) : ℝ := 
  real.sqrt ((A.fst - F.fst)^2 + (A.snd - F.snd)^2)

-- The proof statement
theorem length_AF_is_4 (F : ℝ × ℝ) (A : ℝ × ℝ) 
  (h_focus : is_focus F)
  (h_point : is_point_on_parabola A)
  (h_midpoint : midpoint_condition A F) :
  distance A F = 4 :=
sorry

end length_AF_is_4_l508_508468


namespace sequence_is_increasing_l508_508477

variable (a_n : ℕ → ℝ)

def sequence_positive_numbers (a_n : ℕ → ℝ) : Prop :=
∀ n, 0 < a_n n

def sequence_condition (a_n : ℕ → ℝ) : Prop :=
∀ n, a_n (n + 1) = 2 * a_n n

theorem sequence_is_increasing 
  (h1 : sequence_positive_numbers a_n) 
  (h2 : sequence_condition a_n) : 
  ∀ n, a_n (n + 1) > a_n n :=
by
  sorry

end sequence_is_increasing_l508_508477


namespace compute_expression_l508_508980

variable {p q r : ℝ}

def polynomial (x : ℝ) := x^3 - 7*x^2 + 11*x - 14 = 0

axiom roots_of_polynomial (hp : polynomial p) (hq : polynomial q) (hr : polynomial r)
    : p + q + r = 7 ∧ pq + qr + rp = 11 ∧ pqr = 14

theorem compute_expression (hp : polynomial p) (hq : polynomial q) (hr : polynomial r) :
  roots_of_polynomial hp hq hr →
  (pq/q + qr/p + rp/q) = -75/14 :=
by
  sorry

end compute_expression_l508_508980


namespace woman_completion_days_l508_508681

variable (M W : ℚ)
variable (work_days_man work_days_total : ℚ)

-- Given conditions
def condition1 : Prop :=
  (10 * M + 15 * W) * 7 = 1

def condition2 : Prop :=
  M * 100 = 1

-- To prove
def one_woman_days : ℚ := 350

theorem woman_completion_days (h1 : condition1 M W) (h2 : condition2 M) :
  1 / W = one_woman_days :=
by
  sorry

end woman_completion_days_l508_508681


namespace exist_irreducible_polynomial_with_specified_roots_l508_508780

noncomputable def is_irreducible_polynomial (f : ℚ[X, Y]) : Prop :=
  irreducible f

noncomputable def has_only_specified_roots (f : ℚ[X, Y]) : Prop :=
  (f.eval₂ (polynomial.C : ℚ →+* ℚ[X]) X 1 = 0) ∧
  (f.eval₂ (polynomial.C : ℚ →+* ℚ[X]) 1 X = 0) ∧
  (f.eval₂ (polynomial.C : ℚ →+* ℚ[X]) X (-1) = 0) ∧
  (f.eval₂ (polynomial.C : ℚ →+* ℚ[X]) (-1) X = 0) ∧
  ∀ (x y : ℚ), (x^2 + y^2 = 1) → (f.eval₂ (polynomial.C : ℚ →+* ℚ[X]) x y = 0) → ((x, y) = (0,1) ∨ (x, y) = (1,0) ∨ (x, y) = (0,-1) ∨ (x, y) = (-1,0))

theorem exist_irreducible_polynomial_with_specified_roots : ∃ f : ℚ[X, Y], is_irreducible_polynomial f ∧ has_only_specified_roots f :=
sorry

end exist_irreducible_polynomial_with_specified_roots_l508_508780


namespace sum_of_first_twelve_terms_arithmetic_sequence_l508_508437

-- Definitions
def a (a1 d : ℝ) (n : ℕ) : ℝ := a1 + (n - 1) * d

def Sn (a1 d : ℝ) (n : ℕ) : ℝ := (n / 2) * (2 * a1 + (n - 1) * d)

-- Main Statement
theorem sum_of_first_twelve_terms_arithmetic_sequence  (a1 d : ℝ) 
  (h1 : a a1 d 5 = 1) (h2 : a a1 d 17 = 18) :
  Sn a1 d 12 = 37.5 :=
sorry

end sum_of_first_twelve_terms_arithmetic_sequence_l508_508437


namespace solve_f_x_greater_than_1_range_of_f_on_0_2_l508_508457

-- Proof statement for question (1)
theorem solve_f_x_greater_than_1 (f : ℝ → ℝ) 
  (h : ∀ x, f(x + 1) + f(x - 1) = 2 * x^2 - 2 * x) : 
  {x : ℝ | f(x) > 1} = {x : ℝ | x < -1 ∨ x > 2} :=
sorry

-- Proof statement for question (2)
theorem range_of_f_on_0_2 (f : ℝ → ℝ)
  (h : ∀ x, f(x + 1) + f(x - 1) = 2 * x^2 - 2 * x) :
  Set.range (λ x, f x) ∩ Set.Icc (0 : ℝ) 2 = Set.Icc (-5/4 : ℝ) 1 :=
sorry

end solve_f_x_greater_than_1_range_of_f_on_0_2_l508_508457


namespace matrix_scales_vector_by_3_l508_508420

variables {R : Type*} [CommRing R]
variables (v : Fin 3 → R)

def N : Matrix (Fin 3) (Fin 3) R :=
  !![![3, 0, 0], ![0, 3, 0], ![0, 0, 3]]

theorem matrix_scales_vector_by_3 :
  (N) ⬝ (v) = 3 • v :=
by
  sorry

end matrix_scales_vector_by_3_l508_508420


namespace polynomial_division_l508_508042

theorem polynomial_division (a b c : ℕ) :
  (12 * a^4 * b^3 * c) / (-4 * a^3 * b^2) = -3 * a * b * c :=
by
  sorry

end polynomial_division_l508_508042


namespace total_cases_sold_is_correct_l508_508301

-- Define the customer groups and their respective number of cases bought
def n1 : ℕ := 8
def k1 : ℕ := 3
def n2 : ℕ := 4
def k2 : ℕ := 2
def n3 : ℕ := 8
def k3 : ℕ := 1

-- Define the total number of cases sold
def total_cases_sold : ℕ := n1 * k1 + n2 * k2 + n3 * k3

-- The proof statement that the total cases sold is 40
theorem total_cases_sold_is_correct : total_cases_sold = 40 := by
  -- Proof content will be provided here.
  sorry

end total_cases_sold_is_correct_l508_508301


namespace set_intersection_complement_l508_508232

open Set

theorem set_intersection_complement (U : Set ℝ) (A : Set ℝ) (B : Set ℝ) :
  U = univ ∧ A = {x : ℝ | x > 3 ∨ x < -1} ∧ B = {x : ℝ | x > 0} →
  (U \ A) ∩ B = {x : ℝ | 0 < x ∧ x ≤ 3} :=
by
  intro h
  obtain ⟨hU, hA, hB⟩ := h
  rw [hU, hA, hB]
  sorry

end set_intersection_complement_l508_508232


namespace missing_number_value_l508_508071

def evaluate_expression (x : ℝ) : ℝ :=
  abs (9 - 8 * (x - 12)) - abs (5 - 11)

theorem missing_number_value :
  ∃ (x : ℝ), evaluate_expression x = 75 ∧ x = 3 :=
by {
  use 3,
  split,
  {
    dsimp [evaluate_expression],
    norm_num,
  },
  {
    refl,
  }
}

end missing_number_value_l508_508071


namespace smallest_gcd_value_l508_508511

theorem smallest_gcd_value (m n : ℕ) (hmn : Nat.gcd m n = 15) (hm : m > 0) (hn : n > 0) : Nat.gcd (14 * m) (20 * n) = 30 := 
sorry

end smallest_gcd_value_l508_508511


namespace select_epsilons_l508_508970

variable {n : ℕ} 
variable {a : Fin n → ℝ} -- Fin n is a type representing {0, 1, ..., n-1}

theorem select_epsilons : 
  ∃ (ε : Fin n → ℝ) (hε : ∀ i, ε i = 1 ∨ ε i = -1), 
  (∑ i, a i)^2 + (∑ i, ε i * a i)^2 ≤ (n + 1) * ∑ i, (a i)^2 := 
sorry

end select_epsilons_l508_508970


namespace eternal_number_max_FM_l508_508900

theorem eternal_number_max_FM
  (a b c d : ℕ)
  (h1 : b + c + d = 12)
  (h2 : a = b - d)
  (h3 : (1000 * a + 100 * b + 10 * c + d) - (1000 * b + 100 * a + 10 * d + c) = 81 * (100 * a - 100 * b + c - d))
  (h4 : ∃ k : ℤ, F(M) = 9 * k) :
  ∃ a b c d : ℕ, 100 * (b - d) - 100 * b + 12 - b - 102 * d = 9 := sorry

end eternal_number_max_FM_l508_508900


namespace percent_increase_l508_508390

/-- Problem statement: Given (1/2)x = 1, prove that the percentage increase from 1/2 to x is 300%. -/
theorem percent_increase (x : ℝ) (h : (1/2) * x = 1) : 
  ((x - (1/2)) / (1/2)) * 100 = 300 := 
by
  sorry

end percent_increase_l508_508390


namespace area_AEF_eq_67_5_l508_508546

noncomputable def area_triangle_abc : ℝ := 180

def is_midpoint (P A B : Point) : Prop :=
∃ s: ℝ, 0 < s ∧ s < 1 ∧ P = A + s * (B - A)

variables (A B C F D E : Point)
  (hF : is_midpoint F B C)
  (hD : is_midpoint D A B)
  (hE : is_midpoint E D B)
  (h_area_ABC : area_triangle_abc = 180)

theorem area_AEF_eq_67_5 : area (triangle A E F) = 67.5 := sorry

end area_AEF_eq_67_5_l508_508546


namespace sum_integers_l508_508341

variables {n : ℕ} (a : Fin n → ℤ)
def p (i : Fin n) : ℤ := ∏ j in (Finset.univ.erase i), (a i - a j)

theorem sum_integers (k : ℕ) : 
  (∑ i in Finset.univ, a i ^ k / p a i) ∈ ℤ := sorry

end sum_integers_l508_508341


namespace largest_possible_cos_a_l508_508565

theorem largest_possible_cos_a (a b c : ℝ) (h1 : Real.sin a = Real.cot b) 
  (h2 : Real.sin b = Real.cot c) (h3 : Real.sin c = Real.cot a) : 
  Real.cos a ≤ Real.sqrt ((3 - Real.sqrt 5) / 2) :=
by sorry

end largest_possible_cos_a_l508_508565


namespace arbitrary_tetrahedron_volume_perimeter_equality_case_condition_l508_508999

-- Definitions/constants for the problem
variables (V P : ℝ)

-- The main theorem to prove
theorem arbitrary_tetrahedron_volume_perimeter (V P : ℝ) :
  V <= (8 * real.sqrt 3 / 27) * P^3 :=
sorry

-- The equality case conditions
theorem equality_case_condition (V P : ℝ) :
  V = (8 * real.sqrt 3 / 27) * P^3 ↔ (/* specific conditions */ true) :=
sorry

end arbitrary_tetrahedron_volume_perimeter_equality_case_condition_l508_508999


namespace difference_of_sums_l508_508658

-- Definitions of the involved sequences.
def even_sum (n : ℕ) := n / 2 * (2 + 2 * n)
def primes_sum (n : ℕ) := ∑ i in (Finset.range n).filter Nat.prime, i

-- Variables
variables 
  (n : ℕ)
  (S_primes : ℕ)
  (hn : n = 3005)
  (S_primes_def : S_primes = primes_sum 3005)

-- Theorem statement
theorem difference_of_sums :
  even_sum 3005 - S_primes = 9039030 - S_primes := by
  sorry

end difference_of_sums_l508_508658


namespace three_digit_numbers_square_ends_in_1001_l508_508080

theorem three_digit_numbers_square_ends_in_1001 (n : ℕ) :
  100 ≤ n ∧ n < 1000 ∧ n^2 % 10000 = 1001 → n = 501 ∨ n = 749 :=
by
  intro h
  sorry

end three_digit_numbers_square_ends_in_1001_l508_508080


namespace trig_identity_l508_508885

theorem trig_identity 
  (α : ℝ) 
  (h : Real.tan α = 1 / 3) : 
  Real.cos α ^ 2 + Real.cos (π / 2 + 2 * α) = 3 / 10 :=
sorry

end trig_identity_l508_508885


namespace parabola_point_coordinates_l508_508166

theorem parabola_point_coordinates (x y : ℝ) (h_parabola : y^2 = 8 * x) 
    (h_distance_focus : (x + 2)^2 + y^2 = 81) : 
    (x = 7 ∧ y = 2 * Real.sqrt 14) ∨ (x = 7 ∧ y = -2 * Real.sqrt 14) :=
by {
  -- Proof will be inserted here
  sorry
}

end parabola_point_coordinates_l508_508166


namespace length_is_25_percent_less_than_breadth_l508_508618

theorem length_is_25_percent_less_than_breadth :
  ∀ (A B : ℝ), 
  A = 360 → 
  B = 21.908902300206645 → 
  (let L := 360 / 21.908902300206645 in
   let D := B - L in
   let P := (D / B) * 100 in
  P = 25) :=
by
  intros A B hA hB
  rw [hA, hB]
  let L := 360 / 21.908902300206645
  let D := B - L
  let P := (D / B) * 100
  sorry

end length_is_25_percent_less_than_breadth_l508_508618


namespace min_k_intersects_circle_l508_508102

def circle_eq (x y : ℝ) := (x + 2)^2 + y^2 = 4
def line_eq (x y k : ℝ) := k * x - y - 2 * k = 0

theorem min_k_intersects_circle :
  (∀ k : ℝ, (∃ x y : ℝ, circle_eq x y ∧ line_eq x y k) → k ≥ - (Real.sqrt 3) / 3) :=
sorry

end min_k_intersects_circle_l508_508102


namespace integers_between_sqrt8_and_sqrt78_l508_508149

theorem integers_between_sqrt8_and_sqrt78 : 
  ∃ (n : ℕ), n = 6 ∧ ∀ (x : ℤ), (3 ≤ x ∧ x ≤ 8) ↔ (√8 < x ∧ x < √78) :=
by
  sorry

end integers_between_sqrt8_and_sqrt78_l508_508149


namespace find_a_l508_508817

open Real

-- Declaring the main theorem based on the translated problem
theorem find_a (a m y1 y2 : ℝ) (h1 : y1 + y2 = 8 * m)
(h2 : y1 * y2 = -8 * a) 
(h3 : ∀ m', a ≠ 0 → (1 / ((m^2 + 1) * y1^2) + 1 / ((m^2 + 1) * y2^2)) = (1 / ((4 * a^2) * (m^2 + 1))) * (4 * m^2 + a)) :
  a = 4 :=
sorry

end find_a_l508_508817


namespace range_of_a_l508_508860

noncomputable def f (a x : ℝ) : ℝ := a * Real.log x - 2 * x

theorem range_of_a 
  (a : ℝ) 
  (h : ∀ x : ℝ, 1 < x → 2 * a * Real.log x ≤ 2 * x^2 + f a (2 * x - 1)) :
  a ≤ 2 :=
sorry

end range_of_a_l508_508860


namespace Brown_is_criminal_l508_508038

def Person := { Brown, Jones, Smith : Prop }

-- Defining the statements
def Brown_statement_1 (Brown : Prop) : Prop := ¬ Brown
def Brown_statement_2 (Jones : Prop) : Prop := ¬ Jones
def Jones_statement_1 (Brown : Prop) : Prop := ¬ Brown
def Jones_statement_2 (Smith : Prop) : Prop := Smith
def Smith_statement_1 (Bronwich : Prop) : Prop := Brown
def Smith_statement_2 (Smith : Prop) : Prop := ¬ Smith

-- Defining the truth/lie conditions
def lies_twice_truth_twice_lies_once (Brown : Prop) (Jones : Prop) (Smith : Prop) : Prop :=
  -- Brown lies twice
  (¬ Brown_statement_1 Brown ∧ ¬ Brown_statement_2 Jones)
  -- Jones lies once and tells the truth once
  ∨ (Jones_statement_1 Brown ∧ ¬ Jones_statement_2 Smith ∧ ¬ Jones_statement_1 Brown ∧ Jones_statement_2 Smith)
  -- Smith tells the truth twice
  ∧ (Smith_statement_1 Brown ∧ Smith_statement_2 Smith)

-- The theorem to be proven
theorem Brown_is_criminal (Brown Jones Smith : Prop)
  (h_conditions : lies_twice_truth_twice_lies_once Brown Jones Smith) : Brown :=
  sorry

end Brown_is_criminal_l508_508038


namespace product_floor_ceil_sequence_l508_508404

theorem product_floor_ceil_sequence :
  (Int.floor (-6 - 0.5) * Int.ceil (6 + 0.5) *
   Int.floor (-5 - 0.5) * Int.ceil (5 + 0.5) *
   Int.floor (-4 - 0.5) * Int.ceil (4 + 0.5) *
   Int.floor (-3 - 0.5) * Int.ceil (3 + 0.5) *
   Int.floor (-2 - 0.5) * Int.ceil (2 + 0.5) *
   Int.floor (-1 - 0.5) * Int.ceil (1 + 0.5) *
   Int.floor (-0.5) * Int.ceil (0.5)) = -25401600 := sorry

end product_floor_ceil_sequence_l508_508404


namespace range_f_1_range_m_l508_508851

noncomputable def f (x : ℝ) : ℝ := (Real.log x / Real.log 2 - 2) * (Real.log x / (2 * Real.log 2) - 1/2)

theorem range_f_1 (x : ℝ) (h1 : 2 ≤ x) (h2 : x ≤ 4) : 
  -1/8 ≤ f x ∧ f x ≤ 0 :=
sorry

theorem range_m (m : ℝ) (x : ℝ) (h1 : 4 ≤ x) (h2 : x ≤ 16) (h3 : f x ≥ m * Real.log x / Real.log 2) :
  m ≤ 0 :=
sorry

end range_f_1_range_m_l508_508851


namespace insphere_volume_of_tetrahedron_SABC_l508_508189

noncomputable def tetrahedron_volume_of_insphere_radius (r : ℝ) : ℝ :=
  (4 * ℝ.pi * (r ^ 3)) / 3

theorem insphere_volume_of_tetrahedron_SABC :
  (SA SB SC BC : ℝ) (hc : SA = SB ∧ SB = SC ∧ SC = √21) (hb : BC = 6)
  (hproj : ∀ A : point, projection A (plane_of (S B C)) = orthocenter (triangle_of S B C)) :
  ∃ V : ℝ, V = tetrahedron_volume_of_insphere_radius 1 :=
  sorry

end insphere_volume_of_tetrahedron_SABC_l508_508189


namespace candy_distribution_proof_l508_508741

theorem candy_distribution_proof :
  ∀ (candy_total Kate Robert Bill Mary : ℕ),
  candy_total = 20 →
  Kate = 4 →
  Robert = Kate + 2 →
  Bill = Mary - 6 →
  Kate = Bill + 2 →
  Mary > Robert →
  (Mary - Robert = 2) :=
by
  intros candy_total Kate Robert Bill Mary h1 h2 h3 h4 h5 h6
  sorry

end candy_distribution_proof_l508_508741


namespace general_term_l508_508130

def a (n : ℕ) : ℕ := 2^n
def b (n : ℕ) : ℕ := 3^n
def c (n : ℕ) : ℕ := ∑ i in finset.range(n+1), a i * b (n-i)

theorem general_term :
  ∀ n : ℕ, c n = 6 * (3^n - 2^n) :=
by
  sorry

end general_term_l508_508130


namespace integers_between_sqrt8_and_sqrt78_l508_508148

theorem integers_between_sqrt8_and_sqrt78 : 
  ∃ (n : ℕ), n = 6 ∧ ∀ (x : ℤ), (3 ≤ x ∧ x ≤ 8) ↔ (√8 < x ∧ x < √78) :=
by
  sorry

end integers_between_sqrt8_and_sqrt78_l508_508148


namespace ac_values_l508_508514

theorem ac_values (a c : ℝ) (k : ℤ) :
  (∀ x : ℝ, 2 * sin (3 * x) = a * cos (3 * x + c)) → 
  ∃ k : ℤ, ac = (4 * k - 1) * π :=
sorry

end ac_values_l508_508514


namespace points_on_axis_not_in_quadrants_l508_508329

-- Define what it means for a point to be on the coordinate axis
def on_coordinate_axis (p : ℝ × ℝ) : Prop :=
  p.1 = 0 ∨ p.2 = 0

-- Define what it means to be in a specific quadrant
def in_quadrant (p : ℝ × ℝ) (q : ℕ) : Prop :=
  match q with
  | 1 => p.1 > 0 ∧ p.2 > 0
  | 2 => p.1 < 0 ∧ p.2 > 0
  | 3 => p.1 < 0 ∧ p.2 < 0
  | 4 => p.1 > 0 ∧ p.2 < 0
  | _ => false

-- Points on the coordinate axis do not belong to any quadrant
theorem points_on_axis_not_in_quadrants : ∀ (p : ℝ × ℝ), on_coordinate_axis p → ¬(in_quadrant p 1 ∨ in_quadrant p 2 ∨ in_quadrant p 3 ∨ in_quadrant p 4) :=
by
  intro p h
  cases p with x y
  cases h
  case inl =>
    intro H
    cases H
    case inl => simp at *; exact h.1
    case inr => cases H
    case inl => simp at *; exact h.1
    case inr => cases H
    case inl => simp at *; exact h.1
    case inr => simp at *; exact h.1
  case inr =>
    intro H
    cases H
    case inl => cases H
    case inl => simp at *; exact h.1
    case inr => cases H
    case inl => simp at *; exact h.1
    case inr => cases H
    case inl => simp at *; exact h.1
    case inr => simp at *; exact h.1

end points_on_axis_not_in_quadrants_l508_508329


namespace OneEmptyBox_NoBoxEmptyNoCompleteMatch_AtLeastTwoMatches_l508_508233

def combination (n k : ℕ) : ℕ := Nat.choose n k
def arrangement (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

theorem OneEmptyBox (n : ℕ) (hn : n = 5) : (combination 5 2) * (arrangement 5 5) = 1200 := by
  sorry

theorem NoBoxEmptyNoCompleteMatch (n : ℕ) (hn : n = 5) : (arrangement 5 5) - 1 = 119 := by
  sorry

theorem AtLeastTwoMatches (n : ℕ) (hn : n = 5) : (arrangement 5 5) - (combination 5 1 * 9 + 44) = 31 := by
  sorry

end OneEmptyBox_NoBoxEmptyNoCompleteMatch_AtLeastTwoMatches_l508_508233


namespace jenny_popcorn_kernels_l508_508953

theorem jenny_popcorn_kernels
  (d : ℕ) (k : ℕ) (t : ℕ) (eaten_ratio : ℚ)
  (kernel_drop_rate : k = 1)
  (distance_to_school : d = 5000)
  (drop_rate : ∀ distance, distance / 25 = kernel_drop_rate * distance / 25)
  (squirrel_eats : eaten_ratio = 1 / 4)
  : t = (d / 25) - ((d / 25) * eaten_ratio) :=
by
  sorry

end jenny_popcorn_kernels_l508_508953


namespace power_function_monotonic_l508_508284

theorem power_function_monotonic (m : ℝ) :
  2 * m^2 + m > 0 ∧ m > 0 → m = 1 / 2 := 
by
  intro h
  sorry

end power_function_monotonic_l508_508284


namespace count_linear_eqs_l508_508277

-- Define each equation as conditions
def eq1 (x y : ℝ) := 3 * x - y = 2
def eq2 (x : ℝ) := x + 1 / x + 2 = 0
def eq3 (x : ℝ) := x^2 - 2 * x - 3 = 0
def eq4 (x : ℝ) := x = 0
def eq5 (x : ℝ) := 3 * x - 1 ≥ 5
def eq6 (x : ℝ) := 1 / 2 * x = 1 / 2
def eq7 (x : ℝ) := (2 * x + 1) / 3 = 1 / 6 * x

-- Proof statement: there are exactly 3 linear equations
theorem count_linear_eqs : 
  (∃ x y, eq1 x y) ∧ eq4 0 ∧ (∃ x, eq6 x) ∧ (∃ x, eq7 x) ∧ 
  ¬ (∃ x, eq2 x) ∧ ¬ (∃ x, eq3 x) ∧ ¬ (∃ x, eq5 x) → 
  3 = 3 :=
sorry

end count_linear_eqs_l508_508277


namespace adjusted_volume_bowling_ball_l508_508350

noncomputable def bowling_ball_diameter : ℝ := 40
noncomputable def hole1_diameter : ℝ := 5
noncomputable def hole1_depth : ℝ := 10
noncomputable def hole2_diameter : ℝ := 4
noncomputable def hole2_depth : ℝ := 12
noncomputable def expected_adjusted_volume : ℝ := 10556.17 * Real.pi

theorem adjusted_volume_bowling_ball :
  let radius := bowling_ball_diameter / 2
  let volume_ball := (4 / 3) * Real.pi * radius^3
  let hole1_radius := hole1_diameter / 2
  let hole1_volume := Real.pi * hole1_radius^2 * hole1_depth
  let hole2_radius := hole2_diameter / 2
  let hole2_volume := Real.pi * hole2_radius^2 * hole2_depth
  let adjusted_volume := volume_ball - hole1_volume - hole2_volume
  adjusted_volume = expected_adjusted_volume :=
by
  sorry

end adjusted_volume_bowling_ball_l508_508350


namespace count_standard_sequences_2668_l508_508161

def is_standard_sequence (a : ℕ) (n : ℕ) : Prop :=
  ∃ (a₁ : ℕ), a₁ + a₁ + 1 + ... + a₁ + (n-1) = a

theorem count_standard_sequences_2668 : 
  ∃ (count : ℕ), count = 6 ∧ ∀ (n : ℕ), n > 2 → is_standard_sequence 2668 n → count = 6 :=
by
  sorry

end count_standard_sequences_2668_l508_508161


namespace percentage_error_square_area_l508_508332

theorem percentage_error_square_area (s : ℝ) (h : s > 0) :
  let s' := (1.02 * s)
  let actual_area := s^2
  let measured_area := s'^2
  let error_area := measured_area - actual_area
  let percentage_error := (error_area / actual_area) * 100
  percentage_error = 4.04 := 
sorry

end percentage_error_square_area_l508_508332


namespace regular_nonagon_diagonal_intersections_l508_508029

theorem regular_nonagon_diagonal_intersections : 
  let n := 9 in 
  let num_diagonals := (n * (n - 3)) / 2 in 
  let num_intersection_points := Nat.choose n 4 in 
  num_intersection_points = 126 :=
by
  sorry

end regular_nonagon_diagonal_intersections_l508_508029


namespace variance_five_scores_l508_508375

theorem variance_five_scores (a : ℝ)
  (h : (121 + 127 + 123 + a + 125) / 5 = 124) :
  let mean := 124 in
  let scores := [121, 127, 123, a, 125] in
  let sq_diff_sum := (121 - mean) ^ 2 + (127 - mean) ^ 2 + (123 - mean) ^ 2 + (a - mean) ^ 2 + (125 - mean) ^ 2 in
  let variance := sq_diff_sum / 5 in
  variance = 4 := 
begin
  sorry
end

end variance_five_scores_l508_508375


namespace area_OME_is_3_l508_508177

-- Given definitions from the problem
def O : Point := sorry
def A : Point := sorry
def B : Point := sorry
def M : Point := {x ∈ line_segment A B | x is midpoint}  -- midpoint of AB
def E : Point := sorry

-- Defining given conditions
def radius_OA : ℝ := 5  -- radius of the circle
def chord_AB : ℝ := 8  -- length of chord AB
def OM_perpendicular_to_AB : is_perpendicular O M A B := sorry  -- OM ⊥ AB
def M_midpoint_of_AB : is_midpoint M A B := sorry  -- M is midpoint of AB
def EM_perpendicular_to_OA : is_perpendicular E M O A := sorry  -- EM ⊥ OA

-- The proof problem (final area calculation)
theorem area_OME_is_3 (O A B M E : Point) 
    (h1 : dist O A = 5) 
    (h2 : dist A B = 8) 
    (h3 : is_perpendicular O M A B) 
    (h4 : is_midpoint M A B) 
    (h5 : is_perpendicular E M O A):
    area (triangle O M E) = 3 := by
  sorry

end area_OME_is_3_l508_508177


namespace total_volume_of_water_l508_508689

-- Define the conditions
def volume_of_hemisphere : ℕ := 4
def number_of_hemispheres : ℕ := 2734

-- Define the total volume
def total_volume : ℕ := volume_of_hemisphere * number_of_hemispheres

-- State the theorem
theorem total_volume_of_water : total_volume = 10936 :=
by
  -- Proof placeholder
  sorry

end total_volume_of_water_l508_508689


namespace log_identity_l508_508894

theorem log_identity (a b : ℝ) (h1 : a = Real.log 144 / Real.log 4) (h2 : b = Real.log 12 / Real.log 2) : a = b := 
by
  sorry

end log_identity_l508_508894


namespace range_f_a2_min_value_f_3_l508_508128

noncomputable def f (x a : ℝ) : ℝ := 4 * x^2 - 4 * a * x + (a^2 - 2 * a + 2)

theorem range_f_a2 : ∀ x ∈ Icc 1 2, f x 2 ∈ Icc (-2 : ℝ) 2 :=
by
  sorry

theorem min_value_f_3 (h1 : ∃ x ∈ Icc 0 2, f x a = 3):
  a = 1 - Real.sqrt 2 ∨ a = 5 + Real.sqrt 10 :=
by
  sorry

end range_f_a2_min_value_f_3_l508_508128


namespace nearest_integer_ratio_l508_508519

variable (a b : ℝ)

-- Given condition and constraints
def condition : Prop := (a > b) ∧ (b > 0) ∧ (a + b) / 2 = 3 * Real.sqrt (a * b)

-- Main statement to prove
theorem nearest_integer_ratio (h : condition a b) : Int.floor (a / b) = 34 ∨ Int.floor (a / b) = 33 := sorry

end nearest_integer_ratio_l508_508519


namespace fixed_point_intersection_l508_508487

variables {p a b : ℝ}
variable hp : p ≠ 0
variables ha hb : a ≠ 0 ∧ b ≠ 0
variable hpb2 : b^2 ≠ 2 * p * a

-- Definition of the parabola
def on_parabola (x y : ℝ) : Prop :=
  y^2 = 2 * p * x

-- Definitions of points A and B
def A := (a, b)
def B := (-a, 0)

-- Intersection points on the parabola
variables {y0 y1 y2 : ℝ}
def M := (y0^2 / (2 * p), y0)
def M1 := (y1^2 / (2 * p), y1)
def M2 := (y2^2 / (2 * p), y2)

-- Definition of existence and distinctness of M1 and M2
variables (M : on_parabola (y0^2 / (2 * p)) y0)
variable (M1_exists : ∃ y1, on_parabola (y1^2 / (2 * p)) y1)
variable (M2_exists : ∃ y2, on_parabola (y2^2 / (2 * p)) y2)
variable hM1M2 : M1 ≠ M2

-- The proof statement
theorem fixed_point_intersection :
  ∀ M M1 M2, on_parabola (y0^2 / (2 * p)) y0 →
              (∃ y1, on_parabola (y1^2 / (2 * p)) y1) →
              (∃ y2, on_parabola (y2^2 / (2 * p)) y2) →
              M1 ≠ M2 →
              ∀ x y, x = (a, 2 * p * a / b) → 
              passes_through_fixed_point M1 M2 (a, 2 * p * a / b) :=
sorry

end fixed_point_intersection_l508_508487


namespace c_range_l508_508455

variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)
variable (c : ℝ)

def sequence_positive : Prop := ∀ n, a n > 0
def sequence_sum {n : ℕ} : Prop := S n = ∑ i in finset.range n, a (i + 1)
def sequence_def {n : ℕ} : Prop := a n = 2 * real.sqrt (S n) - 1
def arithmetic_condition (m k n : ℕ) : Prop := m < k ∧ k < n ∧ 2 * k = m + n

theorem c_range {c : ℝ} 
  (h1 : sequence_positive a)
  (h2 : ∀ n, S n = ∑ i in finset.range n, a (i + 1))
  (h3 : ∀ n, a n = 2 * real.sqrt (S n) - 1)
  (h4 : ∀ m k n : ℕ, arithmetic_condition m k n → S m + S n > c * S k) :
  c ≤ 2 := sorry

end c_range_l508_508455


namespace polynomial_at_five_l508_508222

theorem polynomial_at_five (P : ℝ → ℝ) 
  (hP_degree : ∃ (a b c d : ℝ), ∀ x : ℝ, P x = a*x^3 + b*x^2 + c*x + d)
  (hP1 : P 1 = 1 / 3)
  (hP2 : P 2 = 1 / 7)
  (hP3 : P 3 = 1 / 13)
  (hP4 : P 4 = 1 / 21) :
  P 5 = -3 / 91 :=
sorry

end polynomial_at_five_l508_508222


namespace number_of_students_in_both_ball_and_track_l508_508063

variable (total studentsSwim studentsTrack studentsBall bothSwimTrack bothSwimBall bothTrackBall : ℕ)
variable (noAllThree : Prop)

theorem number_of_students_in_both_ball_and_track
  (h_total : total = 26)
  (h_swim : studentsSwim = 15)
  (h_track : studentsTrack = 8)
  (h_ball : studentsBall = 14)
  (h_both_swim_track : bothSwimTrack = 3)
  (h_both_swim_ball : bothSwimBall = 3)
  (h_no_all_three : noAllThree) :
  bothTrackBall = 5 := by
  sorry

end number_of_students_in_both_ball_and_track_l508_508063


namespace quadratic_inequality_solution_set_l508_508089

theorem quadratic_inequality_solution_set :
  {x : ℝ | x * (x - 2) > 0} = {x : ℝ | x < 0} ∪ {x : ℝ | x > 2} :=
by
  sorry

end quadratic_inequality_solution_set_l508_508089


namespace probability_of_BEI3_is_zero_l508_508175

def isVowelOrDigit (s : Char) : Prop :=
  (s ∈ ['A', 'E', 'I', 'O', 'U']) ∨ (s.isDigit)

def isNonVowel (s : Char) : Prop :=
  s ∈ ['B', 'C', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Y', 'Z']

def isHexDigit (s : Char) : Prop :=
  s.isDigit ∨ s ∈ ['A', 'B', 'C', 'D', 'E', 'F']

noncomputable def numPossiblePlates : Nat :=
  13 * 21 * 20 * 16

theorem probability_of_BEI3_is_zero :
    ∃ (totalPlates : Nat), 
    (totalPlates = numPossiblePlates) ∧
    ¬(isVowelOrDigit 'B') →
    (1 : ℚ) / (totalPlates : ℚ) = 0 :=
by
  sorry

end probability_of_BEI3_is_zero_l508_508175


namespace course_selection_schemes_count_l508_508710

-- Define the total number of courses
def total_courses : ℕ := 8

-- Define the number of courses to choose
def courses_to_choose : ℕ := 5

-- Define the two specific courses, Course A and Course B
def courseA := 1
def courseB := 2

-- Define the combination function
def C (n k : ℕ) : ℕ := Nat.choose n k

-- Define the count when neither Course A nor Course B is selected
def case1 : ℕ := C 6 5

-- Define the count when exactly one of Course A or Course B is selected
def case2 : ℕ := C 2 1 * C 6 4

-- Combining both cases
theorem course_selection_schemes_count : case1 + case2 = 36 :=
by
  -- These would be replaced with actual combination calculations.
  sorry

end course_selection_schemes_count_l508_508710


namespace range_of_a_l508_508485

def f (a x : ℝ) : ℝ := log x - 2 * a * x + 2 * a

def g (a x : ℝ) : ℝ := x * (log x - 2 * a * x + 2 * a) + a * x² - x

theorem range_of_a 
  (a : ℝ) (h : ∀ x : ℝ, 0 < x → g a x ≤ g a 1) (h1 : a ∈ ℝ) : a > 1 / 2 :=
by
  sorry

end range_of_a_l508_508485


namespace rounding_to_two_decimal_places_l508_508736

theorem rounding_to_two_decimal_places (x : ℝ) (h : x = 2.7982) : 
  Real.approx x 0.01 = 2.80 :=
by 
  rw h
  exact Real.approximation_rounding_method 2.7982 0.01
  sorry

end rounding_to_two_decimal_places_l508_508736


namespace problem1_l508_508757

theorem problem1 : (-1)^2023 + (-1/2)^(-2) - (3.14 - Real.pi)^0 = 2 := by
  sorry

end problem1_l508_508757


namespace find_number_l508_508344

theorem find_number (x : ℝ) : ((1.5 * x) / 7 = 271.07142857142856) → x = 1265 :=
by
  sorry

end find_number_l508_508344
