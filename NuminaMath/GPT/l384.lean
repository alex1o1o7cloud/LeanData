import Complex
import Mathlib
import Mathlib.Algebra
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.ConicSections
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Order
import Mathlib.Algebra.Order.Ring
import Mathlib.Algebra.Polynomial
import Mathlib.Algebra.QuadraticDiscriminant
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Calculus.Monotone
import Mathlib.Analysis.Convex.Basic
import Mathlib.Analysis.Geometry
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.Volume.ConesAndCylinders
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.SimpleGraph.Basic
import Mathlib.Combinatorics.SimpleGraph.Connectivity
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Complex.Exponential
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Digits
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Rat.Default
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.NumberTheory.Logarithms
import Mathlib.ProbTheory.Probability
import Mathlib.Probability.Kernel.Basic
import Mathlib.Probability.ProbMeasure
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.LibrarySearch
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Pigeonhole
import Mathlib.Topology.EuclideanSpace.Basic

namespace terminating_decimals_l384_384698

theorem terminating_decimals (n : ℤ) (h : 1 ≤ n ∧ n ≤ 180) :
  (∃ k : ℤ, k = 20 ∧ ∀ n, 1 ≤ n ∧ n ≤ 180 → (∃ m, m * 180 = n * (2^2 * 5))) := by
  sorry

end terminating_decimals_l384_384698


namespace sum_of_coordinates_l384_384537

-- Define the given conditions as hypotheses
def isThreeUnitsFromLine (x y : ℝ) : Prop := y = 18 ∨ y = 12
def isTenUnitsFromPoint (x y : ℝ) : Prop := (x - 5)^2 + (y - 15)^2 = 100

-- We aim to prove the sum of the coordinates of the points satisfying these conditions
theorem sum_of_coordinates (x1 x2 x3 x4 y1 y2 y3 y4 : ℝ) 
  (h1 : isThreeUnitsFromLine x1 y1 ∧ isTenUnitsFromPoint x1 y1)
  (h2 : isThreeUnitsFromLine x2 y2 ∧ isTenUnitsFromPoint x2 y2)
  (h3 : isThreeUnitsFromLine x3 y3 ∧ isTenUnitsFromPoint x3 y3)
  (h4 : isThreeUnitsFromLine x4 y4 ∧ isTenUnitsFromPoint x4 y4) :
  x1 + x2 + x3 + x4 + y1 + y2 + y3 + y4 = 50 :=
  sorry

end sum_of_coordinates_l384_384537


namespace shortest_piece_length_l384_384583

variable (x : ℝ) -- the length of the third piece.

-- Conditions
def total_length (x : ℝ) : Prop := x + 3 * x + 6 * x = 138
def second_piece_length (x : ℝ) : Prop := 3 * x
def first_piece_length (x : ℝ) : Prop := 2 * (3 * x)

theorem shortest_piece_length (x : ℝ) (h_total : total_length x) : 
  x = 13.8 :=
by
  sorry

end shortest_piece_length_l384_384583


namespace sum_of_prime_factors_143_l384_384645

theorem sum_of_prime_factors_143 : 
  (∃ (p q : ℕ), prime p ∧ prime q ∧ 143 = p * q ∧ p + q = 24) :=
sorry

end sum_of_prime_factors_143_l384_384645


namespace expected_number_of_failures_l384_384590

noncomputable def expected_failures (N n m : ℕ) (p : ℝ) := N * (Nat.choose n m) * (p^m) * ((1 - p)^(n - m))

theorem expected_number_of_failures (N n m : ℕ) (p : ℝ) (h_prob : 0 ≤ p ∧ p ≤ 1) :
  (∑ i in Finset.range N, 1 : ℝ) * (Nat.choose n m : ℝ) * p^m * (1 - p)^(n - m) = expected_failures N n m p :=
by
  sorry

end expected_number_of_failures_l384_384590


namespace determine_g1_l384_384470

variable (g : ℝ → ℝ)
variable (h : ∀ x y : ℝ, g (g x + y) = g (x + y) + x * g y - x^2 * y - x^3 + 1)

theorem determine_g1 : g 1 = 2 := sorry

end determine_g1_l384_384470


namespace bisect_ih_angle_bhc_l384_384074

variables (O A B C D E F G H I : Type*)
variables [geometry O] [geometry A] [geometry B] [geometry C] [geometry D]
variables [is_centroid H (triangle A B C)]
variables [angle_bisector A B C F]
variables [perpendicular (line CE) (line AB) E]
variables [perpendicular (line BD) (line AC) D]
variables [circumcircle (triangle A E C) O]
variables [intersec_af_circle A F (circumcircle (triangle A E C)) F]
variables [circumcircle (triangle A D E) (intersects G)]
variables [intersec_gf_bc G F I]

theorem bisect_ih_angle_bhc
  (AF_bisects_BAC : angle_bisector (line AF) (angle BAC))
  (H_centroid_of_ABC : is_centroid H (triangle ABC))
  (CE_perp_AB_at_E : perpendicular (line CE) (line AB) E)
  (BD_perp_AC_at_D : perpendicular (line BD) (line AC) D)
  (circumcircle_ADE_intersects_circumcircle_AEC_at_G : circumcircle_intersect (circumcircle (triangle ADE)) (circumcircle (triangle AEC)) G)
  (GF_intersects_BC_at_I : line_intersect (line GF) BC I) :
  bisects (line IH) (angle BHC) :=
sorry

end bisect_ih_angle_bhc_l384_384074


namespace action_figures_added_l384_384453

theorem action_figures_added
  (initially_had : ℕ)
  (removed : ℕ)
  (currently_have : ℕ) :
  (initially_had = 15) →
  (removed = 7) →
  (currently_have = 10) →
  ∃ (A : ℕ), (initially_had - removed + A = currently_have) ∧ (A = 2) :=
by
  intros h_initially_had h_removed h_currently_have
  use (currently_have - (initially_had - removed))
  split
  · rw [h_initially_had, h_removed, h_currently_have]
    norm_num
  sorry

end action_figures_added_l384_384453


namespace terminating_decimals_l384_384708

theorem terminating_decimals (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 180) :
  (∃ k : ℕ, k = 20 ∧ ∀ m : ℕ, 1 ≤ m ∧ m ≤ 180 → (decimal_terminates (m / 180) ↔ m % 9 = 0)) :=
by {
  sorry
}

def decimal_terminates (x : ℚ) : Prop :=
  sorry

end terminating_decimals_l384_384708


namespace square_area_eq_l384_384282

-- Define the side length of the square and the diagonal relationship
variables (s : ℝ) (h : s * Real.sqrt 2 = s + 1)

-- State the theorem to solve
theorem square_area_eq :
  s * Real.sqrt 2 = s + 1 → (s ^ 2 = 3 + 2 * Real.sqrt 2) :=
by
  -- Assume the given condition
  intro h
  -- Insert proof steps here, analysis follows the provided solution steps.
  sorry

end square_area_eq_l384_384282


namespace prime_add_eq_2001_l384_384002

theorem prime_add_eq_2001 (a b : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) (h : a^2 + b = 2003) : a + b = 2001 :=
sorry

end prime_add_eq_2001_l384_384002


namespace sum_of_odd_divisors_of_72_l384_384222

def is_odd (n : ℕ) : Prop := n % 2 = 1

def divisors (n : ℕ) : List ℕ :=
  List.filter (λ d, n % d = 0) (List.range (n + 1))

def odd_divisors_sum (n : ℕ) : ℕ :=
  (divisors n).filter is_odd |>.sum

theorem sum_of_odd_divisors_of_72 : odd_divisors_sum 72 = 13 := 
  sorry

end sum_of_odd_divisors_of_72_l384_384222


namespace rose_part_payment_l384_384130

-- Defining the conditions
def total_cost (T : ℝ) := 0.95 * T = 5700
def part_payment (x : ℝ) (T : ℝ) := x = 0.05 * T

-- The proof problem: Prove that the part payment Rose made is $300
theorem rose_part_payment : ∃ T x, total_cost T ∧ part_payment x T ∧ x = 300 :=
by
  sorry

end rose_part_payment_l384_384130


namespace factor_t_squared_minus_81_l384_384319

theorem factor_t_squared_minus_81 (t : ℂ) : (t^2 - 81) = (t - 9) * (t + 9) := 
by
  -- We apply the identity a^2 - b^2 = (a - b) * (a + b)
  let a := t
  let b := 9
  have eq : t^2 - 81 = a^2 - b^2 := by sorry
  rw [eq]
  exact (mul_sub_mul_add_eq_sq_sub_sq a b).symm
  -- Concluding the proof
  sorry -- skipping detailed proof steps for now

end factor_t_squared_minus_81_l384_384319


namespace tangent_sum_l384_384882

-- Given a regular hexagon ABCDEF
structure RegularHexagon (A B C D E F : Point) : Prop :=
(is_regular : true) -- Placeholder to define a regular hexagon

-- Let P be a point within the hexagon
structure InteriorPoint (ABCDEF : Triangle) (P : Point) : Prop :=
(is_interior : true) -- Placeholder to define a point inside the hexagon

-- Let A, B, C, D, E, F be points such that they form a regular hexagon.
variables {A B C D E F P : Point}

-- Then we ought to ensure the lean theorem includes the tangent equation.
theorem tangent_sum (h1 : RegularHexagon A B C D E F) (h2 : InteriorPoint A B C D E F P) :
  tan (angle A P D) + tan (angle B P E) = tan (angle C P F) := sorry

end tangent_sum_l384_384882


namespace alice_meeting_distance_l384_384554

noncomputable def distanceAliceWalks (t : ℝ) : ℝ :=
  6 * t

theorem alice_meeting_distance :
  ∃ t : ℝ, 
    distanceAliceWalks t = 
      (900 * Real.sqrt 2 - Real.sqrt 630000) / 11 ∧
    (5 * t) ^ 2 =
      (6 * t) ^ 2 + 150 ^ 2 - 2 * 6 * t * 150 * Real.cos (Real.pi / 4) :=
sorry

end alice_meeting_distance_l384_384554


namespace find_vector_Q_l384_384449

variables (A B C : Type) [AddCommGroup A] [Module ℚ A] [AddCommGroup B] [Module ℚ B] [AddCommGroup C] [Module ℚ C]

variables {α β γ : Type} [linear_map_space α A] [linear_map_space β B] [linear_map_space γ C]

def vecQ (A B C : Type) [AddCommGroup A] [Module ℚ A] [AddCommGroup B] [Module ℚ B] [AddCommGroup C] [Module ℚ C] :=
  let D := (3/5 : ℚ) • A + (2/5 : ℚ) • C in
  let G := (2/5 : ℚ) • A + (3/5 : ℚ) • B in
  let H := (1/3 : ℚ) • B + (2/3 : ℚ) • C in
  ∃ Q : A × B × C, Q = ((12/35 : ℚ) • A + (9/35 : ℚ) • B + (14/35 : ℚ) • C)

theorem find_vector_Q (A B C : Type) [AddCommGroup A] [Module ℚ A] [AddCommGroup B] [Module ℚ B] [AddCommGroup C] [Module ℚ C] :
  vecQ A B C :=
  sorry

end find_vector_Q_l384_384449


namespace find_fifth_term_of_sequence_l384_384810

def sequence (n : ℕ) : ℤ :=
  Nat.recOn n
    3                       -- for n = 0
    (λ n ih, Nat.casesOn n  -- for n = 1
      6
      (λ n, ih.snd - ih.fst) -- for n >= 2
    )

theorem find_fifth_term_of_sequence :
  sequence 4 = -6 := -- Remember that in Lean, sequences are usually 0-indexed
by
  -- sorry, skipping the proof as per the instruction
  sorry

end find_fifth_term_of_sequence_l384_384810


namespace area_greater_than_perimeter_greater_than_l384_384975

structure Triangle :=
(base : ℝ)
(ac : ℝ)
(cb : ℝ)
(angle_acb : ℝ)

def area (T : Triangle) : ℝ := 
  0.5 * T.base * T.ac * T.cb * Math.sin T.angle_acb

def perimeter (T : Triangle) : ℝ :=
  T.base + T.ac + T.cb

variables {C1 C2 : Triangle}
-- Hypotheses
-- Common base
(h_base : C1.base = C2.base)
-- Same angle
(h_angle : C1.angle_acb = C2.angle_acb)
-- Condition on side lengths
(h_condition : abs (C1.ac - C1.cb) < abs (C2.ac - C2.cb))

-- Proof goals
theorem area_greater_than :
  area C1 > area C2 := 
sorry

theorem perimeter_greater_than :
  perimeter C1 > perimeter C2 := 
sorry

end area_greater_than_perimeter_greater_than_l384_384975


namespace quadrilateral_perimeter_is_12_sqrt_5_l384_384304

-- Definitions from the conditions
def isQuadrilateralInscribedInRectangle : Prop := sorry
def twoSidesAreSidesOfRectangle : Prop := sorry
def oppositeVerticesConnectedByDiagonal : Prop := sorry
def rectangleLengthTwiceWidth (l w : ℝ) : Prop := l = 2 * w
def rectangleDiagonal (d : ℝ): Prop := d = 10

-- The width and length of the rectangle
def width (w : ℝ) := w
def length (l : ℝ) := 2 * w

-- The with the length l and width w
noncomputable def perimeterQuadrilateral (l w : ℝ) := 2 * l + 2 * w

-- The statement to prove
theorem quadrilateral_perimeter_is_12_sqrt_5 (w : ℝ) (l : ℝ)
    (h_quad_inscribed : isQuadrilateralInscribedInRectangle)
    (h_two_sides_rectangle : twoSidesAreSidesOfRectangle)
    (h_opposite_vertices_diagonal : oppositeVerticesConnectedByDiagonal)
    (h_length_twice_width : rectangleLengthTwiceWidth l w)
    (h_diagonal : rectangleDiagonal 10) :
    perimeterQuadrilateral l w = 12 * Real.sqrt 5 := sorry

end quadrilateral_perimeter_is_12_sqrt_5_l384_384304


namespace remainder_of_T_mod_11_l384_384476

-- Definitions and conditions from the problem
def n : ℕ := sorry
def T : ℕ := (10 * (10^n - 1)) / 81 - n / 9

-- Theorem statement
theorem remainder_of_T_mod_11 (n : ℕ) (T : ℕ) (hT : T = (10 * (10^n - 1)) / 81 - n / 9) : 
  T % 11 = (nat.floor((n + 1) / 2) % 11) :=
sorry

end remainder_of_T_mod_11_l384_384476


namespace area_of_semicircle_l384_384605

-- Definitions based on conditions
def rectangle_width : ℝ := 2
def rectangle_length : ℝ := 3
def circle_diameter : ℝ := 5 -- given by the Pythagorean theorem from the solution steps
def circle_radius : ℝ := circle_diameter / 2

-- Area calculation of the semicircle
def semicircle_area : ℝ := (π * (circle_radius ^ 2)) / 2

-- Statement to prove
theorem area_of_semicircle :
  semicircle_area = 25 * π / 8 :=
by
  sorry

end area_of_semicircle_l384_384605


namespace arithmetic_sequence_a2_a9_l384_384376

theorem arithmetic_sequence_a2_a9 (a : ℕ → ℚ) (d : ℚ)
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_sum : a 5 + a 6 = 12) :
  a 2 + a 9 = 12 :=
sorry

end arithmetic_sequence_a2_a9_l384_384376


namespace ratio_of_x_to_y_l384_384644

theorem ratio_of_x_to_y (x y : ℝ) (h : (8 * x - 5 * y) / (11 * x - 3 * y) = 4 / 7) : x / y = 23 / 12 := 
by
  sorry

end ratio_of_x_to_y_l384_384644


namespace terminating_decimals_l384_384711

theorem terminating_decimals (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 180) :
  (∃ k : ℕ, k = 20 ∧ ∀ m : ℕ, 1 ≤ m ∧ m ≤ 180 → (decimal_terminates (m / 180) ↔ m % 9 = 0)) :=
by {
  sorry
}

def decimal_terminates (x : ℚ) : Prop :=
  sorry

end terminating_decimals_l384_384711


namespace neg_p_equiv_exists_leq_l384_384394

-- Define the given proposition p
def p : Prop := ∀ x : ℝ, x^2 - x + 1 > 0

-- State the equivalence we need to prove
theorem neg_p_equiv_exists_leq :
  ¬ p ↔ ∃ x : ℝ, x^2 - x + 1 ≤ 0 :=
by {
  sorry  -- Proof is skipped as per instructions
}

end neg_p_equiv_exists_leq_l384_384394


namespace perpendicular_bisector_AB_parallel_line_through_P_l384_384397

-- Define the concept of a Point in a 2D plane
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define the concept of a Line in 2D
structure Line :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

-- Define specific points A, B, and P
def A := Point.mk 6 (-6)
def B := Point.mk 2 2
def P := Point.mk 2 (-3)

-- Define the properties of the perpendicular bisector of segment AB
def is_perpendicular_bisector (L : Line) (A B : Point) :=
  (L.a * A.x + L.b * A.y + L.c = 0) ∧
  (L.a * B.x + L.b * B.y + L.c = 0) ∧
  L.b ≠ 0 ∧
  (L.a = 2 ∗ L.b)

-- Define the properties of the line passing through P and parallel to AB
def is_parallel_through_point (L : Line) (P A B : Point) :=
  (L.a * P.x + L.b * P.y + L.c = 0) ∧
  (L.a * (B.x - A.x) + L.b * (B.y - A.y) = 0) ∧
  (A.x - B.x) * (P.y - B.y) = (A.y - B.y) * (P.x - B.x)

-- Problem: Proving the properties for the given lines
theorem perpendicular_bisector_AB :
  ∃ L : Line, is_perpendicular_bisector L A B ∧ L = {a := 1, b := -2, c := -8} :=
sorry

theorem parallel_line_through_P :
  ∃ L : Line, is_parallel_through_point L P A B ∧ L = {a := 2, b := 1, c := -1} :=
sorry

end perpendicular_bisector_AB_parallel_line_through_P_l384_384397


namespace rhombus_area_range_l384_384806

theorem rhombus_area_range:
  (∀ (l : Line) (O : Circle),
    l.equation = (λ x, sqrt (3) * x + 4) ∧
    O.equation = (λ x y, x^2 + y^2 = r^2) ∧
      1 < r ∧ r < 2 ∧
      (∃ A B, A ∈ l ∧ B ∈ l) ∧
      (∃ C D, C ∈ O ∧ D ∈ O)
    → (exists S, S ∈ Set.Union (Set.Ioo 0 (3 * sqrt(3) / 2)) (Set.Ioo (3 * sqrt(3) / 2) (6 * sqrt(3)))))
:= sorry

end rhombus_area_range_l384_384806


namespace remainder_8357_to_8361_div_9_l384_384335

theorem remainder_8357_to_8361_div_9 :
  (8357 + 8358 + 8359 + 8360 + 8361) % 9 = 3 := 
by
  sorry

end remainder_8357_to_8361_div_9_l384_384335


namespace points_on_parabola_l384_384755

theorem points_on_parabola (s : ℝ) : ∃ (u v : ℝ), u = 3^s - 4 ∧ v = 9^s - 7 * 3^s - 2 ∧ v = u^2 + u - 14 :=
by
  sorry

end points_on_parabola_l384_384755


namespace maximize_partial_sum_l384_384366

variable {a : ℕ → ℝ} -- Assuming the sequence is represented by a function from natural numbers to real numbers.

theorem maximize_partial_sum (h1 : a 7 + a 8 + a 9 > 0) (h2 : a 7 + a 10 < 0) :
  ∀ n, (∑ i in Finset.range (n + 1), a i) ≤ (∑ i in Finset.range 9, a i) := by
  sorry

end maximize_partial_sum_l384_384366


namespace range_of_m_l384_384347

theorem range_of_m (m : ℝ) : (∀ x1 : ℝ, ∃ x2 ∈ set.Icc (3 : ℝ) 4, x1^2 + x1 * x2 + x2^2 ≥ 2 * x1 + m * x2 + 3) →
  m ≤ 3 := sorry

end range_of_m_l384_384347


namespace triangle_BC_value_l384_384856

theorem triangle_BC_value (B C A : ℝ) (AB AC BC : ℝ) 
  (hB : B = 45) 
  (hAB : AB = 100)
  (hAC : AC = 100)
  (h_deg : A ≠ 0) :
  BC = 100 * Real.sqrt 2 := 
by 
  sorry

end triangle_BC_value_l384_384856


namespace largest_prime_factor_of_1023_l384_384568

/-- The largest prime factor of 1023 is 31. -/
theorem largest_prime_factor_of_1023 : ∃ p : ℕ, nat.prime p ∧ ∃ q : ℕ, 1023 = p * q ∧ (∀ r : ℕ, nat.prime r ∧ r ∣ 1023 → r ≤ p) := 
sorry

end largest_prime_factor_of_1023_l384_384568


namespace solve_fraction_equation_l384_384924

theorem solve_fraction_equation (x : ℝ) :
  (1 / (x + 10) + 1 / (x + 8) = 1 / (x + 11) + 1 / (x + 7)) ↔ x = -9 :=
by {
  sorry
}

end solve_fraction_equation_l384_384924


namespace matching_balls_l384_384192

-- Definitions capturing the problem conditions
def Ball (c : Color) := Fin 2022
def Box (c : Color) := Ball c × Ball c
def black_boxes := List (Box Color.Black)
def white_boxes := List (Box Color.White)

-- Problem statement
theorem matching_balls :
  ∀ (black_boxes : List (Box Color.Black))
    (white_boxes : List (Box Color.White))
    (h_black : black_boxes.length = 1011)
    (h_white : white_boxes.length = 1011),
    ∃ (chosen_black : List (Ball Color.Black))
      (chosen_white : List (Ball Color.White)),
      chosen_black.length = 1011 ∧
      chosen_white.length = 1011 ∧
      (chosen_black ++ chosen_white).perm (List.finRange 2022) :=
sorry

end matching_balls_l384_384192


namespace math_problem_l384_384149

theorem math_problem (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : (4*x + y) / (x - 4*y) = -3) : 
  (x + 4*y) / (4*x - y) = 39 / 37 :=
by
  sorry

end math_problem_l384_384149


namespace terminating_decimal_fraction_count_l384_384694

theorem terminating_decimal_fraction_count :
  (finset.filter (λ n : ℕ, 1 ≤ n ∧ n ≤ 180 ∧ 90 ∣ n) (finset.range 181)).card = 20 :=
by sorry

end terminating_decimal_fraction_count_l384_384694


namespace envelope_hypocycloid_epicycloid_num_return_points_l384_384999

noncomputable theory

-- Definition of the points
def pointA (ϕ : ℝ) : ℂ := Complex.exp (Complex.I * ϕ)
def pointB (k ϕ : ℝ) : ℂ := Complex.exp (Complex.I * k * ϕ)

-- Hypothesis for k
variables {k : ℝ} (hk : k ≠ 0 ∧ k ≠ 1 ∧ k ≠ -1)

-- Part (a): Prove envelope is hypocycloid or epicycloid
theorem envelope_hypocycloid_epicycloid (ϕ : ℝ) : 
  ∃ (C : ℝ → ℂ), ∀ φ, C φ = 
    let A := pointA φ in
    let B := pointB k φ in
    (B + k * A) / (1 + k) :=
sorry

-- Part (b): Number of return points
theorem num_return_points (k : ℝ) (hk : k ≠ 0 ∧ k ≠ 1 ∧ k ≠ -1) : 
  ∃ n, n = |k - 1| :=
sorry

end envelope_hypocycloid_epicycloid_num_return_points_l384_384999


namespace sum_of_digits_l384_384569

theorem sum_of_digits (n : ℕ) (h : n = 3 ^ 2003 * 5 ^ 2005 * 2) : 
  (sum_of_digits n) = 7 :=
sorry

end sum_of_digits_l384_384569


namespace sum_of_selected_numbers_l384_384524

theorem sum_of_selected_numbers (n : ℕ) : 
  let grid := (λ i j , (i-1)*n + j)
  in S = ∑ i in finRange n, S_row i + ∑ j in finRange n, S_col j → 
  S = (n * (n^2 + 1)) / 2 :=
by
  sorry


end sum_of_selected_numbers_l384_384524


namespace mryak_bryak_problem_l384_384969

variable (m b : ℚ)

theorem mryak_bryak_problem
  (h1 : 3 * m = 5 * b + 10)
  (h2 : 6 * m = 8 * b + 31) :
  7 * m - 9 * b = 38 := sorry

end mryak_bryak_problem_l384_384969


namespace root_conditions_l384_384831

theorem root_conditions (a : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ |x1^2 - 5 * x1| = a ∧ |x2^2 - 5 * x2| = a) ↔ (a = 0 ∨ a > 25 / 4) := 
by 
  sorry

end root_conditions_l384_384831


namespace wheat_acres_l384_384509

theorem wheat_acres (x y : ℤ) 
  (h1 : x + y = 4500) 
  (h2 : 42 * x + 35 * y = 165200) : 
  y = 3400 :=
sorry

end wheat_acres_l384_384509


namespace contrapositive_equivalence_l384_384225

theorem contrapositive_equivalence (P Q : Prop) : (P → Q) ↔ (¬ Q → ¬ P) :=
by sorry

end contrapositive_equivalence_l384_384225


namespace sqrt_of_six_l384_384152

theorem sqrt_of_six : Real.sqrt 6 = Real.sqrt 6 := by
  sorry

end sqrt_of_six_l384_384152


namespace three_digit_number_reduce_to_zero_l384_384285

theorem three_digit_number_reduce_to_zero (n : ℕ) (h1 : 100 ≤ n) (h2 : n < 1000) :
  (iterate (λ x, x - (x / 100 + (x % 100) / 10 + x % 10)) 100 n) = 0 :=
by
  sorry

end three_digit_number_reduce_to_zero_l384_384285


namespace probability_triangle_forming_in_decagon_l384_384061

theorem probability_triangle_forming_in_decagon :
  ∀ (segments : Finset (Fin 10 × Fin 10)),
    segments.card = 45 →
    let valid_triangle_probability := 373 / 495
    in ∃ (selected_segments : Finset (Fin 10 × Fin 10)),
      selected_segments.card = 3 ∧
      ∀ (a b c : ℝ),
        a ∈ selected_segments.map (λ (p : Fin 10 × Fin 10), 2 * Real.sin (↑(p.2 - p.1) * Real.pi / 10)) →
        b ∈ selected_segments.map (λ (p : Fin 10 × Fin 10), 2 * Real.sin (↑(p.2 - p.1) * Real.pi / 10)) →
        c ∈ selected_segments.map (λ (p : Fin 10 × Fin 10), 2 * Real.sin (↑(p.2 - p.1) * Real.pi / 10)) →
        (a + b > c) ∧ (a + c > b) ∧ (b + c > a) → 
        (selected_segments.card.choose 3) * valid_triangle_probability = (373 : ℝ) / 495 := sorry

end probability_triangle_forming_in_decagon_l384_384061


namespace area_of_ABHED_is_54_l384_384072

noncomputable def area_ABHED (DA DE : ℝ) (AB EF : ℝ) (BH : ℝ) (congruent : Bool) : ℝ :=
  if congruent then 96 + (7 * Real.sqrt 89 / 2) - (8 * Real.sqrt 73 / 2) else 0

theorem area_of_ABHED_is_54 : 
  area_ABHED 8 8 12 12 7 true = 54 :=
by
  sorry

end area_of_ABHED_is_54_l384_384072


namespace terminating_decimals_l384_384706

theorem terminating_decimals (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 180) :
  (∃ k : ℕ, k = 20 ∧ ∀ m : ℕ, 1 ≤ m ∧ m ≤ 180 → (decimal_terminates (m / 180) ↔ m % 9 = 0)) :=
by {
  sorry
}

def decimal_terminates (x : ℚ) : Prop :=
  sorry

end terminating_decimals_l384_384706


namespace triangle_XYZ_b_h_find_base_min_area_triangle_l384_384475

-- (a) Confirm b, h for given coordinates.
theorem triangle_XYZ_b_h (X Y Z : ℝ × ℝ) (h b : ℝ) :
  X = (2, 4) →
  Y = (0, 0) →
  Z = (4, 0) →
  h = 4 →
  b = 4 → (X, Y, Z, h, b) :=
by
  intros
  sorry

-- (b) Given h = 3 and s = 2, find base b.
theorem find_base (h s b : ℝ) :
  h = 3 →
  s = 2 →
  b = 8 / 3 :=
by
  intros
  sorry

-- (c) Given the area of square is 2017, find the minimum area of triangle XYZ.
theorem min_area_triangle (s triangle_area : ℝ) :
  s^2 = 2017 →
  triangle_area = 36 :=
by
  intros
  sorry

end triangle_XYZ_b_h_find_base_min_area_triangle_l384_384475


namespace Caitlin_Sara_weight_l384_384291

variable (A C S : ℕ)

theorem Caitlin_Sara_weight 
  (h1 : A + C = 95) 
  (h2 : A = S + 8) : 
  C + S = 87 := by
  sorry

end Caitlin_Sara_weight_l384_384291


namespace num_terminating_decimals_l384_384681

theorem num_terminating_decimals (n : ℕ) (h1 : 1 ≤ n ∧ n ≤ 180) :
  (∃ k : ℕ, k = 20) := by
  have h2 : ∀ n, (1 ≤ n ∧ n ≤ 180) → (∃ k, k = (180 / 9)) := sorry
  have h3 : 180 / 9 = 20 := by norm_num
  use 20
  exact h3

end num_terminating_decimals_l384_384681


namespace arrange_letters_l384_384096

theorem arrange_letters :
  let C := 9! / (2! * 2!)
  let T := 9! / (2! * 2!)
  let S := 9! / (2! * 2! * 2!)
  let M := 6! / (2!)
  in (C - T + S) / M = 126 :=
by
  sorry

end arrange_letters_l384_384096


namespace instantaneous_rate_of_change_at_0_l384_384309

noncomputable def f (x : ℝ) : ℝ := 2 * Real.exp (Real.sin x)

theorem instantaneous_rate_of_change_at_0 : (deriv f 0) = 2 :=
  by
  sorry

end instantaneous_rate_of_change_at_0_l384_384309


namespace segment_equal_twice_inradius_l384_384759

variable {A B C M : Type} [point A] [point B] [point C] [point M]
variable {AD BE CF AI BJ CK : Type} [line AD] [line BE] [line CF] [segment AI] [segment BJ] [segment CK]
variable (t r : ℝ)

-- Assume the necessary geometric relations and properties
axiom is_inside (M : point) (ABC : triangle) : M ∈ interior ABC
axiom is_altitude (A D : point) (BC : line) : B ∈ BC ∧ C ∈ BC ∧ AD ⊥ BC
axiom is_projection (M : point) (line : line) (I : point) : M → I ∈ line ∧ MI ⊥ line
axiom segments_equal (AI : segment) (BJ : segment) (CK : segment) : AI.length = t ∧ BJ.length = t ∧ CK.length = t
axiom area_of_triangle (ABC : triangle) : real
axiom inradius (ABC : triangle) : real

-- Formalize the theorem to be proved
theorem segment_equal_twice_inradius
  (M_in_triangle : is_inside M (triangle A B C))
  (A_altitude : is_altitude A D (line B C))
  (B_altitude : is_altitude B E (line A C))
  (C_altitude : is_altitude C F (line A B))
  (AI_proj : is_projection M (line A D) I)
  (BJ_proj : is_projection M (line B E) J)
  (CK_proj : is_projection M (line C F) K)
  (seg_equal : segments_equal (segment A I) (segment B J) (segment C K))
  (area_ABC : area_of_triangle (triangle A B C) = S)
  (inradius_ABC : inradius (triangle A B C) = r) :
  t = 2 * r :=
by
  sorry

end segment_equal_twice_inradius_l384_384759


namespace terminating_decimal_count_number_of_terminating_decimals_l384_384672

theorem terminating_decimal_count {n : ℕ} (h1 : 1 ≤ n) (h2 : n ≤ 180) :
  (∃ k, n = 9 * k ∧ 1 ≤ 9 * k ∧ 9 * k ≤ 180) ↔ n ∈ (finset.range 21).image (λ k, 9 * k) :=
by sorry

theorem number_of_terminating_decimals :
  (finset.range 21).card = 20 :=
by sorry

end terminating_decimal_count_number_of_terminating_decimals_l384_384672


namespace length_more_than_breadth_l384_384521

theorem length_more_than_breadth (length cost_per_metre total_cost : ℝ) (breadth : ℝ) :
  length = 60 → cost_per_metre = 26.50 → total_cost = 5300 → 
  (total_cost = (2 * length + 2 * breadth) * cost_per_metre) → length - breadth = 20 :=
by
  intros hlength hcost_per_metre htotal_cost hperimeter_cost
  rw [hlength, hcost_per_metre] at hperimeter_cost
  sorry

end length_more_than_breadth_l384_384521


namespace solution_set_f_l384_384388

noncomputable def f (a x : ℝ) : ℝ := log a (x^2) + a ^ (|x|)

theorem solution_set_f (a x : ℝ) (h1 : a > 1) (h2 : f a (-3) < f a 4) :
  f a (x^2 - 2 * x) ≤ f a 3 → x ∈ [-1, 0) ∪ (0, 3] :=
by
  sorry

end solution_set_f_l384_384388


namespace f_value_at_5pi_over_6_l384_384424

noncomputable def f (x ω : ℝ) := 2 * Real.sin (ω * x + (Real.pi / 3))

theorem f_value_at_5pi_over_6
  (ω : ℝ) (ω_pos : ω > 0)
  (α β : ℝ)
  (h1 : f α ω = 2)
  (h2 : f β ω = 0)
  (h3 : Real.sqrt ((α - β)^2 + 4) = Real.sqrt (4 + (Real.pi^2 / 4))) :
  f (5 * Real.pi / 6) ω = -1 := 
sorry

end f_value_at_5pi_over_6_l384_384424


namespace monotonically_increasing_interval_is_neg_infty_neg_3_l384_384955

noncomputable theory

def interval_monotonic_increasing {α : Type*} [linear_order α] (f : α → α) (a b : α) : Prop :=
∀ x y, a ≤ x → y ≤ b → x < y → f x < f y

def log_base_1_2 (x : ℝ) := Real.log x / Real.log (1/2)

def f (x : ℝ) := log_base_1_2 (x^2 - 9)

theorem monotonically_increasing_interval_is_neg_infty_neg_3 :
  interval_monotonic_increasing f (-∞) (-3) :=
sorry

end monotonically_increasing_interval_is_neg_infty_neg_3_l384_384955


namespace exists_five_digit_number_l384_384540

-- Conditions given in the problem
def conditions (n : ℕ) : Prop :=
  n % 1 = 0 ∧
  n % 2 = 1 ∧
  n % 3 = 2 ∧
  n % 4 = 3 ∧
  n % 5 = 4 ∧
  n % 6 = 5 ∧
  n % 7 = 6 ∧
  n % 8 = 7 ∧
  n % 9 = 8 ∧
  n % 10 = 9 ∧
  n % 11 = 10 ∧
  n % 13 = 11

-- The proof statement to show existence of such an 'n'
theorem exists_five_digit_number : ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ conditions n :=
begin
  use 83159,
  split,
  { norm_num },
  split,
  { norm_num },
  { dsimp [conditions], norm_num },
end

end exists_five_digit_number_l384_384540


namespace tan_alpha_plus_beta_max_min_f_l384_384375

variable {α β x : ℝ}

-- Definitions based on conditions
def tan_alpha : ℝ := -1/3
def cos_beta : ℝ := (5:ℝ).sqrt / 5
def sin_beta : ℝ := ((1 - cos_beta^2).sqrt)

-- Conditions: α, β in (0, π)
axiom alpha_in_range : 0 < α ∧ α < real.pi
axiom beta_in_range : 0 < β ∧ β < real.pi

-- Questions and answers
theorem tan_alpha_plus_beta :
  real.tan (α + β) = 1 :=
sorry

theorem max_min_f :
  ∃ max min, max = (5:ℝ).sqrt ∧ min = -((5:ℝ).sqrt) ∧ 
    (∀ x, (√2 * real.sin (x - α) + real.cos (x + β)) ≤ max) ∧
    (∀ x, (√2 * real.sin (x - α) + real.cos (x + β)) ≥ min) :=
sorry

end tan_alpha_plus_beta_max_min_f_l384_384375


namespace max_sum_a_b_c_d_l384_384417

theorem max_sum_a_b_c_d (a c d : ℤ) (b : ℕ) (h1 : a + b = c) (h2 : b + c = d) (h3 : c + d = a) : 
  a + b + c + d = -5 := 
sorry

end max_sum_a_b_c_d_l384_384417


namespace distance_from_B_to_plane_SAC_correct_l384_384363

noncomputable def distance_from_B_to_plane_SAC : Real :=
  let r := 2
  let R := 2 * r
  let SC := R
  let AB := Real.sqrt 3
  let angle_SCA := Real.pi / 6
  let angle_SCB := Real.pi / 6
  let distance := 3 / 2 -- correct answer from solution
  distance
  
-- Theorem to state the problem
theorem distance_from_B_to_plane_SAC_correct :
  ∀ (O S A B C : EuclideanGeometry.Point) (r : Real),
    EuclideanGeometry.Sphere O r ∧ EuclideanGeometry.OnSurface O r S ∧ EuclideanGeometry.OnSurface O r A ∧ 
    EuclideanGeometry.OnSurface O r B ∧ EuclideanGeometry.OnSurface O r C ∧ 
    EuclideanGeometry.Distance S C = 2 * r ∧
    EuclideanGeometry.Distance A B = Real.sqrt 3 ∧
    EuclideanGeometry.Angle S C A = Real.pi / 6 ∧
    EuclideanGeometry.Angle S C B = Real.pi / 6 →
    EuclideanGeometry.DistanceFromPointToPlane B S A C = distance_from_B_to_plane_SAC := 
  by simp; sorry

end distance_from_B_to_plane_SAC_correct_l384_384363


namespace find_p_q_r_l384_384877

open Real

variables {V : Type} [InnerProductSpace V]

-- Defining vectors and scalars according to the problem
variables (a b c d : V) (t p q r : ℝ)

-- Given conditions
axiom orthogonal_unit_vectors : ⟪a, a⟫ = 1 ∧ ⟪c, c⟫ = 1 ∧ ⟪a, c⟫ = 0
axiom b_dot_a_zero : ⟪b, a⟫ = 0
axiom d_definition : d = b + t • a
axiom a_definition : a = p • (b × c) + q • (c × d) + r • (d × b)
axiom a_dot_bc_one : ⟪a, b × c⟫ = 1

-- Question: Prove that p + q + r = 1
theorem find_p_q_r : p + q + r = 1 :=
sorry

end find_p_q_r_l384_384877


namespace factor_t_squared_minus_81_l384_384317

theorem factor_t_squared_minus_81 (t : ℂ) : (t^2 - 81) = (t - 9) * (t + 9) := 
by
  -- We apply the identity a^2 - b^2 = (a - b) * (a + b)
  let a := t
  let b := 9
  have eq : t^2 - 81 = a^2 - b^2 := by sorry
  rw [eq]
  exact (mul_sub_mul_add_eq_sq_sub_sq a b).symm
  -- Concluding the proof
  sorry -- skipping detailed proof steps for now

end factor_t_squared_minus_81_l384_384317


namespace gcd_2210_145_l384_384561

-- defining the constants a and b
def a : ℕ := 2210
def b : ℕ := 145

-- theorem stating that gcd(a, b) = 5
theorem gcd_2210_145 : Nat.gcd a b = 5 :=
sorry

end gcd_2210_145_l384_384561


namespace factor_t_squared_minus_81_l384_384322

theorem factor_t_squared_minus_81 (t : ℝ) : t^2 - 81 = (t - 9) * (t + 9) :=
by
  sorry

end factor_t_squared_minus_81_l384_384322


namespace quadratic_has_two_distinct_real_roots_l384_384185

noncomputable def discriminant (a b c : ℝ) : ℝ := b * b - 4 * a * c

theorem quadratic_has_two_distinct_real_roots :
  (∀ x : ℝ, (x + 1) * (x - 1) = 2 * x + 3) → discriminant 1 (-2) (-4) > 0 :=
by
  intro h
  -- conditions from the problem
  let a := 1
  let b := -2
  let c := -4
  -- use the discriminant function directly with the values
  have delta := discriminant a b c
  show delta > 0
  sorry

end quadratic_has_two_distinct_real_roots_l384_384185


namespace num_solutions_l384_384819

theorem num_solutions (m n : ℕ) (h1 : m > 0) (h2 : n > 0) :
  (∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ 8 / m + 6 / n = 1) ↔ 5 :=
sorry

end num_solutions_l384_384819


namespace three_sides_form_triangle_containing_polygon_l384_384765

theorem three_sides_form_triangle_containing_polygon (N : ℕ) (hN : N ≥ 5) 
  (polygon : list (ℝ × ℝ)) (hconvex : is_convex polygon) : 
  ∃ (a b c: ℝ × ℝ), (a ∈ polygon) ∧ (b ∈ polygon) ∧ (c ∈ polygon) ∧ 
  (triangle_contains_polygon polygon (a, b, c)) :=
sorry

end three_sides_form_triangle_containing_polygon_l384_384765


namespace quadratic_no_real_roots_probability_l384_384626

theorem quadratic_no_real_roots_probability :
  let b_range := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
  let total_pairs := 11 * 11
  let valid_pairs := (List.filter (λ (bc : Int × Int), let b := bc.1; let c := bc.2; c > b^2 / 4) (List.product b_range b_range)).length
  (Rational.mk valid_pairs total_pairs) = 70 / 121 :=
by
  sorry

end quadratic_no_real_roots_probability_l384_384626


namespace quadratic_roots_l384_384961

noncomputable def roots (p q : ℝ) :=
  if h : p = 1 then ⟨1, h, Classical.arbitrary (ℝ), by simp⟩ else
  if h : p = -2 then ⟨-2, -1, -1, by simp [show p = -2 from h]⟩ else
  false.elim (by simp_all)

theorem quadratic_roots (p q : ℝ) :
  (∃ x1 x2, (x1 + x2 = -p ∧ x1 * x2 = q) ∧ (x1 + 1) + (x2 + 1) = p^2 ∧ (x1 + 1) * (x2 + 1) = pq) →
  (p = 1 ∧ ∃ q, true) ∨ (p = -2 ∧ q = -1) :=
by
  intro h
  sorry

end quadratic_roots_l384_384961


namespace sum_of_roots_eq_36_l384_384150

theorem sum_of_roots_eq_36 :
  (∃ x1 x2 x3 : ℝ, (11 - x1) ^ 3 + (13 - x2) ^ 3 = (24 - 2 * x3) ^ 3 ∧ 
  (11 - x2) ^ 3 + (13 - x3) ^ 3 = (24 - 2 * x1) ^ 3 ∧ 
  (11 - x3) ^ 3 + (13 - x1) ^ 3 = (24 - 2 * x2) ^ 3 ∧
  x1 + x2 + x3 = 36) :=
sorry

end sum_of_roots_eq_36_l384_384150


namespace curve_equation_line_equation_intersection_distance_l384_384847

theorem curve_equation (α : ℝ) : 
    ∃ (x y : ℝ), x = 3 * cos α ∧ y = sin α ∧ (x^2 / 9) + y^2 = 1 :=
by
  sorry

theorem line_equation (t : ℝ) : 
    ∃ (x y : ℝ), x = -t + 2 ∧ y = t ∧ x + y - 2 = 0 :=
by
  sorry

theorem intersection_distance (t1 t2 : ℝ) (h1 : t1 + t2 = 2 * Real.sqrt 2 / 5)
  (h2 : t1 * t2 = -1) : 
    |t1 - t2| = 6 * Real.sqrt 3 / 5 :=
by
  sorry

end curve_equation_line_equation_intersection_distance_l384_384847


namespace find_m_values_l384_384825

theorem find_m_values (m : ℕ) : (m - 3) ^ m = 1 ↔ m = 0 ∨ m = 2 ∨ m = 4 := sorry

end find_m_values_l384_384825


namespace factor_of_5_in_20_fact_l384_384230

noncomputable def fact : ℕ → ℕ
| 0       => 1
| (n + 1) => (n + 1) * fact n

theorem factor_of_5_in_20_fact (n : ℕ) (hn : 0 < n) :
  (5^n ∣ fact 20) ∧ ¬ (5^(n + 1) ∣ fact 20) → n = 4 :=
by
  sorry

end factor_of_5_in_20_fact_l384_384230


namespace largest_number_from_digits_l384_384993

def digits := {0, 8, 7}

theorem largest_number_from_digits : ∃ n : ℕ, (∀ d ∈ {0, 8, 7}, d ∈ (digits n)) ∧ n = 870 := 
by
  sorry

end largest_number_from_digits_l384_384993


namespace total_kids_played_l384_384087

-- Definitions based on conditions
def kidsMonday : Nat := 17
def kidsTuesday : Nat := 15
def kidsWednesday : Nat := 2

-- Total kids calculation
def totalKids : Nat := kidsMonday + kidsTuesday + kidsWednesday

-- Theorem to prove
theorem total_kids_played (Julia : Prop) : totalKids = 34 :=
by
  -- Using sorry to skip the proof
  sorry

end total_kids_played_l384_384087


namespace angle_terminal_side_eq_l384_384510

noncomputable def has_same_terminal_side (θ₁ θ₂ : ℝ) : Prop :=
  ∃ k : ℤ, θ₂ = θ₁ + 2 * k * Real.pi

theorem angle_terminal_side_eq (k : ℤ) :
  has_same_terminal_side (- (Real.pi / 3)) (5 * Real.pi / 3) :=
by
  use 1
  sorry

end angle_terminal_side_eq_l384_384510


namespace decreasing_y_as_x_increases_l384_384621

theorem decreasing_y_as_x_increases :
  (∀ x1 x2, x1 < x2 → (-2 * x1 + 1) > (-2 * x2 + 1)) ∧
  ¬ (∀ x1 x2, x1 < x2 → (x1^2 + 1) > (x2^2 + 1)) ∧
  ¬ (∀ x1 x2, x1 < x2 → (-x1^2 + 1) > (-x2^2 + 1)) ∧
  ¬ (∀ x1 x2, x1 < x2 → (2 * x1 + 1) > (2 * x2 + 1)) :=
by
  sorry

end decreasing_y_as_x_increases_l384_384621


namespace perimeter_inequality_area_inequality_l384_384259

noncomputable def convex_n_gon (n : ℕ) (M : Type) := (n ≥ 3) ∧ (n ≥ 4) ∧ convex ℝ M

theorem perimeter_inequality {n : ℕ} (M M' : Type) [convex_n_gon n M] :
  ∀ (P P' : ℝ), P > P' ∧ P' ≥ (1/2) * P :=
sorry

theorem area_inequality {n : ℕ} (M M' : Type) [convex_n_gon n M] :
  n ≥ 4 → ∀ (S S' : ℝ), S > S' ∧ S' ≥ (1/2) * S :=
sorry

end perimeter_inequality_area_inequality_l384_384259


namespace expression_equals_5_l384_384572

theorem expression_equals_5 : (3^2 - 2^2) = 5 := by
  calc
    (3^2 - 2^2) = 5 := by sorry

end expression_equals_5_l384_384572


namespace cos_sin_int_equality_count_l384_384348

theorem cos_sin_int_equality_count : ∃ n : ℕ, n ≤ 500 ∧ (∀ t : ℝ, (complex.exp (complex.I * t))^n = complex.exp (complex.I * (n * t))) ∧ n = 500 :=
by
  let n := 500
  -- Add the necessary conditions and assertions here.
  existsi n
  -- sorry will be replaced by the actual proof in practice.
  sorry

end cos_sin_int_equality_count_l384_384348


namespace geometry_ratios_l384_384858

section triangle_geometry

variables {A B C M K P : Type} [Order A] [Order B] [Order C] [Order M] [Order K] [Order P]

/-- In triangle ABC, let AM be the angle bisector of angle BAC,
BK be the median from B to side AC, and AM be perpendicular to BK at point P.
Given these conditions, we aim to prove the ratios BP : PK = 1 and AP : PM = 3.
-/
theorem geometry_ratios (triangle_ABC : A × B × C)
  (is_angle_bisector_AM : ∠BAC = ∠BAM + ∠CAM)
  (is_median_BK : ∀ (x y : A), BK x y = (x + y) / 2)
  (perpendicular_AM_BK : ∀ (x y : M), ∑ z, AM z = K z)
  (P_intersection : ∀ (x : P), x ∈ (BK ∩ AM).val) :
  (BP : PK = 1) ∧ (AP : PM = 3) :=
by sorry

end triangle_geometry

end geometry_ratios_l384_384858


namespace math_problem_l384_384881

theorem math_problem :
  let x := (3 + Real.sqrt 8) ^ 500
  let n := Real.floor x
  let f := x - n
  x ^ 2 * (1 - f) = (3 + Real.sqrt 8) ^ 500 := by
  sorry

end math_problem_l384_384881


namespace area_ratio_triangle_l384_384012

noncomputable def ratio_of_areas (m : ℝ) : ℝ :=
  (m - 1)^2 / (m^2 + m + 1)

theorem area_ratio_triangle (A B C D E F N M P : Type)
  (m : ℝ)
  (hD : Point D ∈ segment B C)
  (hE : Point E ∈ segment C A)
  (hF : Point F ∈ segment A B)
  (hRatio_AF_FB : 1 / m = segment_length AF / segment_length FB)
  (hRatio_BD_DC : 1 / m = segment_length BD / segment_length DC)
  (hRatio_CE_EA : 1 / m = segment_length CE / segment_length EA)
  (hN : Point N = intersection AD BE)
  (hM : Point M = intersection BE CF)
  (hP : Point P = intersection CF AD)
  : area_ratio M N P A B C = ratio_of_areas m := 
sorry

end area_ratio_triangle_l384_384012


namespace union_sets_eq_l384_384890

-- Definitions of the sets M and N according to the conditions.
def M : Set ℝ := {x | x^2 = x}
def N : Set ℝ := {x | Real.log x ≤ 0}

-- The theorem we want to prove
theorem union_sets_eq :
  (M ∪ N) = Set.Icc 0 1 :=
by
  sorry

end union_sets_eq_l384_384890


namespace books_printed_l384_384086

-- Definitions of the conditions
def book_length := 600
def pages_per_sheet := 8
def total_sheets := 150

-- The theorem to prove
theorem books_printed : (total_sheets * pages_per_sheet / book_length) = 2 := by
  sorry

end books_printed_l384_384086


namespace jill_spent_on_clothing_l384_384900

-- Define the total amount spent excluding taxes, T.
variable (T : ℝ)
-- Define the percentage of T Jill spent on clothing, C.
variable (C : ℝ)

-- Define the conditions based on the problem statement.
def jill_tax_conditions : Prop :=
  let food_percent := 0.20
  let other_items_percent := 0.30
  let clothing_tax := 0.04
  let food_tax := 0
  let other_tax := 0.10
  let total_tax := 0.05
  let food_amount := food_percent * T
  let other_items_amount := other_items_percent * T
  let clothing_amount := C * T
  let clothing_tax_amount := clothing_tax * clothing_amount
  let other_tax_amount := other_tax * other_items_amount
  let total_tax_amount := clothing_tax_amount + food_tax * food_amount + other_tax_amount
  C * T + food_percent * T + other_items_percent * T = T ∧
  total_tax_amount / T = total_tax

-- The goal is to prove that C = 0.50.
theorem jill_spent_on_clothing (h : jill_tax_conditions T C) : C = 0.50 :=
by
  sorry

end jill_spent_on_clothing_l384_384900


namespace arrange_in_ascending_order_l384_384405

theorem arrange_in_ascending_order (x : ℝ) (h1 : -1 < x) (h2 : x < 0) : 5 * x < 0.5 * x ∧ 0.5 * x < 5 - x := by
  sorry

end arrange_in_ascending_order_l384_384405


namespace circle_center_in_third_quadrant_l384_384270

theorem circle_center_in_third_quadrant
  (a b r : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_r : 0 < r) :
  ∃ q, q = 3 ∧ ( (-a, -b) ∈ quadrant q) :=
by
  sorry

end circle_center_in_third_quadrant_l384_384270


namespace permutation_of_two_from_three_l384_384823

-- | Define the number of people.
def n : ℕ := 3

-- | Define the number of people to choose.
def k : ℕ := 2

-- | The expected number of permutations.
def expected_permutations : ℕ := 6

-- | Lean theorem statement.
theorem permutation_of_two_from_three : n.choose k! / (n - k)! = expected_permutations := by
  sorry

end permutation_of_two_from_three_l384_384823


namespace max_knights_in_unfortunate_island_l384_384118

theorem max_knights_in_unfortunate_island (natives : ℕ) (liars_made_a_mistake : ℕ) : 
  natives = 2022 → liars_made_a_mistake = 3 → 
  ∃ (knights : ℕ), knights = 1349 :=
by
  intros h_natives h_mistake
  use 1349
  sorry

end max_knights_in_unfortunate_island_l384_384118


namespace number_of_positive_integers_in_S_l384_384478

theorem number_of_positive_integers_in_S :
  let S := { n : ℕ | n > 1 ∧ ∃ d : ℕ → ℕ, (λ i, d i) = (λ i, d (i + 16)) ∧ 1 / (n : ℝ) = 0.d1 d2 d3 d4 ...}
  let P := 65521
  S.card = 127 :=
by 
  sorry

end number_of_positive_integers_in_S_l384_384478


namespace range_of_a_l384_384000

def set1 : Set ℝ := {x | x ≤ 2}
def set2 (a : ℝ) : Set ℝ := {x | x > a}
variable (a : ℝ)

theorem range_of_a (h : set1 ∪ set2 a = Set.univ) : a ≤ 2 :=
by sorry

end range_of_a_l384_384000


namespace smallest_ratio_l384_384145

-- Define the system of equations as conditions
def eq1 (x y : ℝ) := x^3 + 3 * y^3 = 11
def eq2 (x y : ℝ) := (x^2 * y) + (x * y^2) = 6

-- Define the goal: proving the smallest value of x/y for the solutions (x, y) is -1.31
theorem smallest_ratio (x y : ℝ) (h1 : eq1 x y) (h2 : eq2 x y) :
  ∃ t : ℝ, t = x / y ∧ ∀ t', t' = x / y → t' ≥ -1.31 :=
sorry

end smallest_ratio_l384_384145


namespace prob1_prob2_l384_384485

-- Proof Problem 1: 3001 × 2999 = 8999999 using the difference of squares formula
theorem prob1 : 3001 * 2999 = 8999999 :=
by
  have a := 3000
  have b := 1
  calc
    3001 * 2999 = (a + b) * (a - b) : by rw [add_one_mul_sub_one a b]
    ... = a^2 - b^2 : by rw [mul_sub, sub_mul, mul_add, sub_add_cancel, mul_self, mul_self, add_comm]
    ... = 3000^2 - 1^2 : by { rw [a, b] }
    ... = 9000000 - 1 : by norm_num
    ... = 8999999 : by norm_num

-- Proof Problem 2: (2+1)(2^2+1)(2^4+1)(2^8+1)(2^{16}+1)(2^{32}+1) = 2^64 - 1
theorem prob2 : (2+1)*(2^2+1)*(2^4+1)*(2^8+1)*(2^{16}+1)*(2^{32}+1) = 2^64 - 1 :=
by
  calc
    (2+1)*(2^2+1)*(2^4+1)*(2^8+1)*(2^{16}+1)*(2^{32}+1)
      = (2-1)*(2+1)*(2^2+1)*(2^4+1)*(2^8+1)*(2^{16}+1)*(2^{32}+1) : by simp [two_add_one]
    ... = (2^2 - 1)*(2^2 + 1)*(2^4 + 1)*(2^8 + 1)*(2^{16} + 1)*(2^{32} + 1) : by rw [sub_add_eq_sub_sub]
    ... = (2^4 - 1)*(2^4 + 1)*(2^8 + 1)*(2^{16} + 1)*(2^{32} + 1) : sorry
    ... = (2^8 - 1)*(2^8 + 1)*(2^{16} + 1)*(2^{32} + 1) : sorry
    ... = (2^{16} - 1)*(2^{16} + 1)*(2^{32} + 1) : sorry
    ... = (2^{32} - 1)*(2^{32} + 1) : sorry
    ... = 2^64 - 1 : sorry

end prob1_prob2_l384_384485


namespace color_ngon_condition_l384_384194

def ways_to_color_ngon (n k : ℕ) : ℕ :=
  (k-1)^n + (k-1) * ((-1)^n)

theorem color_ngon_condition (n k : ℕ) (h : k ≥ 2) : 
  ways_to_color_ngon n k = (k-1)^n + (k-1*(-1)^n) := 
sorry

end color_ngon_condition_l384_384194


namespace range_of_p_l384_384770

theorem range_of_p (a : ℕ → ℝ) (S : ℕ → ℝ) (p : ℝ) :
  (∀ n : ℕ, n > 0 → S n = (-1) ^ n * a n + 1/2^n + 2*n - 6) ∧ 
  (∀ n : ℕ, n > 0 → (a (n + 1) - p) * (a n - p) < 0) → 
  p ∈ set.Ioo (-(7/4 : ℝ)) (23/4 : ℝ) :=
sorry

end range_of_p_l384_384770


namespace train_pass_time_l384_384998

def length_of_train : ℝ := 250
def speed_of_train_kmph : ℝ := 68
def speed_of_man_kmph : ℝ := 8

def relative_speed_kmph : ℝ := speed_of_train_kmph - speed_of_man_kmph
def relative_speed_mps : ℝ := (relative_speed_kmph * 1000) / 3600

-- Using this 'noncomputable' modifier because exact division calculations may not always be computable in Lean.
noncomputable def time_to_pass : ℝ := length_of_train / relative_speed_mps

theorem train_pass_time : abs (time_to_pass - 15) < 0.1 := 
by {
  -- Here, you would provide the proof showing that the time_to_pass indeed approximates 15 within an epsilon of 0.1
  sorry
}

end train_pass_time_l384_384998


namespace x_eighteen_l384_384410

theorem x_eighteen (x : ℂ) (hx : x + 1/x = complex.sqrt 3) : x^18 = -1 := 
sorry

end x_eighteen_l384_384410


namespace last_integer_in_sequence_l384_384180

theorem last_integer_in_sequence (a : ℕ) (h_start : a = 524288) (h_seq : ∀ n : ℕ, a / 2^n ∈ ℕ) :
  ∃ k : ℕ, a / 2^k = 1 ∧ (∀ m : ℕ, m > k → a / 2^m ∉ ℕ) :=
by {
  sorry
}

end last_integer_in_sequence_l384_384180


namespace probability_of_A_given_B_l384_384059

-- Definitions of events
def tourist_attractions : List String := ["Pengyuan", "Jiuding Mountain", "Garden Expo Park", "Yunlong Lake", "Pan'an Lake"]

-- Probabilities for each scenario
noncomputable def P_AB : ℝ := 8 / 25
noncomputable def P_B : ℝ := 20 / 25
noncomputable def P_A_given_B : ℝ := 2 / 5

-- Proof statement
theorem probability_of_A_given_B : (P_AB / P_B) = P_A_given_B :=
by
  sorry

end probability_of_A_given_B_l384_384059


namespace total_oysters_eaten_l384_384198

/-- Squido eats 200 oysters -/
def Squido_eats := 200

/-- Crabby eats at least twice as many oysters as Squido -/
def Crabby_eats := 2 * Squido_eats

/-- Total oysters eaten by Squido and Crabby -/
theorem total_oysters_eaten : Squido_eats + Crabby_eats = 600 := 
by
  sorry

end total_oysters_eaten_l384_384198


namespace terminating_decimal_integers_count_l384_384737

/-- For how many integer values of n between 1 and 180 inclusive does the decimal 
representation of n / 180 terminate? We need to calculate the number of integers n such 
that n includes at least 9 as a factor. The answer is 20. -/
theorem terminating_decimal_integers_count :
  { n : ℤ | 1 ≤ n ∧ n ≤ 180 ∧ 9 ∣ n }.to_finset.card = 20 := 
by sorry

end terminating_decimal_integers_count_l384_384737


namespace terminating_decimals_count_l384_384727

theorem terminating_decimals_count : 
  let N := 180 in
  let count_terminating_decimals := (List.range' 1 N).filter (λ n, n % 3 = 0) in
  count_terminating_decimals.length = 60 :=
by
  let N := 180
  let count_terminating_decimals := (List.range' 1 N).filter (λ n, n % 3 = 0)
  sorry

end terminating_decimals_count_l384_384727


namespace larger_square_area_l384_384557

theorem larger_square_area
  (isos_right_triangles : ∀ {a : ℕ} (h : a > 0), isosceles_right_triangle a)
  (small_square_side : ℕ) (h1 : small_square_side = 12)
  (num_triangles_larger : ℕ) (h2 : num_triangles_larger = 18)
  (num_triangles_smaller : ℕ) (h3 : num_triangles_smaller = 16) :
  let smaller_square_area := small_square_side * small_square_side in
  let ratio := num_triangles_larger.toReal / num_triangles_smaller.toReal in
  let expected_larger_square_area := smaller_square_area.toReal * ratio in
  expected_larger_square_area = 162 :=
by
  -- We state the proof is not included here.
  sorry

end larger_square_area_l384_384557


namespace solve_problem_l384_384233

noncomputable def cube (n : ℕ) := n * n * n

noncomputable def eval_expr (a b : ℕ) := -cube a + cube b

noncomputable def factor (a b : ℕ) := a * a - a * b + b * b

theorem solve_problem : 
  eval_expr 666 555 = -124072470 :=
by
  have h1 : factor 666 555 = 381951 := by sorry
  have h2 : eval_expr 666 555 = (- 666 * 666 * 666 + 555 * 555 * 555) := by sorry
  calc
    eval_expr 666 555 
        = - 666 * 666 * 666 + 555 * 555 * 555 : by sorry
    ... = -295408296 + 170953875 : by sorry
    ... = -124454421 : by rfl
    ... + 381951 = -124072470 : by rfl

end solve_problem_l384_384233


namespace cut_squares_to_form_square_l384_384558

theorem cut_squares_to_form_square (a b : ℝ) (h : a < b) : 
    ∃ (x y : ℝ), x^2 + y^2 = (real.sqrt (a^2 + b^2))^2 := 
sorry

end cut_squares_to_form_square_l384_384558


namespace tangent_line_at_2_neg3_l384_384158

noncomputable def tangent_line (x : ℝ) : ℝ := (1 + x) / (1 - x)

theorem tangent_line_at_2_neg3 :
  ∃ m b, ∀ x, (tangent_line x = m * x + b) →
  ∃ y, (2 * x - y - 7 = 0) :=
by
  sorry

end tangent_line_at_2_neg3_l384_384158


namespace smallest_n_l384_384219

theorem smallest_n
  (n : ℕ)
  (h1 : n % 2 = 1)
  (h2 : n % 3 = 1)
  (h3 : n % 4 = 1)
  (h4 : n % 5 = 1)
  (h5 : n % 6 = 1)
  (h6 : n % 7 = 1)
  (h7 : 8 ∣ n) :
  n = 1681 :=
  sorry

end smallest_n_l384_384219


namespace hyperbola_equation_constant_value_hyperbola_l384_384767

def hyperbola (x y : ℝ): Prop :=
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (x / a)^2 - (y / b)^2 = 1

def point_on_hyperbola (x y a b : ℝ): Prop :=
 (x / a)^2 - (y / b)^2 = 1

def eccentricity (a c : ℝ): ℝ := c / a 

noncomputable def constants (a b c : ℝ): Prop :=
  b = 2 * real.sqrt a ∧ c = 4

theorem hyperbola_equation (x y a b c : ℝ) (h1: a > 0) (h2: point_on_hyperbola 4 6 a b) (h3: eccentricity a c = 2) : 
  b = 2 * real.sqrt a ∧ c = 4 ∧ (x = 4 / 3 ∧ y = 0) :=
sorry

theorem constant_value_hyperbola (A B F2x F2y : ℝ) (h1: ∃ l, ∃ m : Prop, line_through l m F2x F2y A B ∧ F2x = 4 ∧ A ≠ V ∧ B ≠ V) :
  ∃ k : ℝ, k = 1 / 3 ∧ ∀ A F2 B,  abs ((1 / (dist (A F2x))) - (1 / (dist (B F2x)))) = k :=
sorry

end hyperbola_equation_constant_value_hyperbola_l384_384767


namespace halfway_fraction_eq_l384_384163

-- Define the fractions
def one_seventh := 1 / 7
def one_fourth := 1 / 4

-- Define the common denominators
def common_denom_1 := 4 / 28
def common_denom_2 := 7 / 28

-- Define the addition of the common denominators
def addition := common_denom_1 + common_denom_2

-- Define the average of the fractions
noncomputable def average := addition / 2

-- State the theorem
theorem halfway_fraction_eq : average = 11 / 56 :=
by
  -- Provide the steps which will be skipped here
  sorry

end halfway_fraction_eq_l384_384163


namespace terminating_decimal_count_number_of_terminating_decimals_l384_384669

theorem terminating_decimal_count {n : ℕ} (h1 : 1 ≤ n) (h2 : n ≤ 180) :
  (∃ k, n = 9 * k ∧ 1 ≤ 9 * k ∧ 9 * k ≤ 180) ↔ n ∈ (finset.range 21).image (λ k, 9 * k) :=
by sorry

theorem number_of_terminating_decimals :
  (finset.range 21).card = 20 :=
by sorry

end terminating_decimal_count_number_of_terminating_decimals_l384_384669


namespace locus_description_l384_384211

noncomputable def A : ℝ × ℝ × ℝ := (-1, 0, 0)
noncomputable def B : ℝ × ℝ × ℝ := (1, 0, 0)

def distance (P Q : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2 + (P.3 - Q.3) ^ 2)

theorem locus_description :
  {P : ℝ × ℝ × ℝ | distance P A - distance P B = 1} = 
  {P : ℝ × ℝ × ℝ | 
    ∃ (Q : ℝ × ℝ), 
    (distance Q (A.1, A.2) - distance Q (B.1, B.2) = 1) ∧ 
    (P = (Q.1, Q.2, Q.3)) } :=
sorry

end locus_description_l384_384211


namespace terminating_decimal_count_number_of_terminating_decimals_l384_384671

theorem terminating_decimal_count {n : ℕ} (h1 : 1 ≤ n) (h2 : n ≤ 180) :
  (∃ k, n = 9 * k ∧ 1 ≤ 9 * k ∧ 9 * k ≤ 180) ↔ n ∈ (finset.range 21).image (λ k, 9 * k) :=
by sorry

theorem number_of_terminating_decimals :
  (finset.range 21).card = 20 :=
by sorry

end terminating_decimal_count_number_of_terminating_decimals_l384_384671


namespace sequence_transform_possible_l384_384196

theorem sequence_transform_possible :
  ∃ f : ℕ → (ℤ → ℤ → ℤ → (ℤ × ℤ × ℤ)), 
    (∀ i, f i = λ x y z, (x + y, -y, z + y)) ∧ 
  (∃ seq : ℕ → list ℤ, seq 0 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10] ∧ 
  (∃ n, seq n = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1] ∧ 
  ∀ i < n, ∃ j < 18, seq (i+1) = list.take j (seq i) ++ (f i (list.nth (seq i) j).get! (list.nth (seq i) (j+1)).get! (list.nth (seq i) (j+2)).get! :: list.drop (j+3) (seq i))) :=
sorry

end sequence_transform_possible_l384_384196


namespace degrees_to_radians_216_l384_384639

theorem degrees_to_radians_216 : (216 / 180 : ℝ) * Real.pi = (6 / 5 : ℝ) * Real.pi := by
  sorry

end degrees_to_radians_216_l384_384639


namespace optimal_tablet_combination_exists_l384_384972

/-- Define the daily vitamin requirement structure --/
structure Vitamins (A B C D : ℕ)

theorem optimal_tablet_combination_exists {x y : ℕ} :
  (∃ (x y : ℕ), 
    (3 * x ≥ 3) ∧ (x + y ≥ 9) ∧ (x + 3 * y ≥ 15) ∧ (2 * y ≥ 2) ∧
    (x + y = 9) ∧ 
    (20 * x + 60 * y = 3) ∧ 
    (x + 2 * y = 12) ∧ 
    (x = 6 ∧ y = 3)) := 
  by
  sorry

end optimal_tablet_combination_exists_l384_384972


namespace range_of_m_l384_384774

-- Definitions and conditions
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f(x) = f(-x)

def condition (f : ℝ → ℝ) : Prop :=
∀ (x₁ x₂ : ℝ), (0 ≤ x₁ ∧ 0 ≤ x₂ ∧ x₁ ≠ x₂) → (x₁ - x₂) * (f x₁ - f x₂) > 0

theorem range_of_m (f : ℝ → ℝ) (h_even : even_function f) (h_cond : condition f)
  (h_comp: f (m + 1) ≥ f 2) : m ∈ Iic (-3) ∪ Ici 1 :=
sorry

end range_of_m_l384_384774


namespace han_xin_troop_min_soldiers_l384_384399

theorem han_xin_troop_min_soldiers (n : ℕ) : 
  (n % 3 = 2) ∧ (n % 5 = 3) ∧ (n % 7 = 4) → n = 53 :=
  sorry

end han_xin_troop_min_soldiers_l384_384399


namespace multiple_of_B_share_l384_384284

theorem multiple_of_B_share (A B C : ℝ) (k : ℝ) 
    (h1 : 3 * A = k * B) 
    (h2 : k * B = 7 * 84) 
    (h3 : C = 84)
    (h4 : A + B + C = 427) :
    k = 4 :=
by
  -- We do not need the detailed proof steps here.
  sorry

end multiple_of_B_share_l384_384284


namespace rectangle_covering_with_dominoes_l384_384903

theorem rectangle_covering_with_dominoes (n m : ℕ) : 
  ∃ (first_layer second_layer : set (ℕ × ℕ)), 
    (∀ (x y : ℕ), (x, y) ∈ first_layer ∨ (x, y) ∈ second_layer) ∧ 
    (∀ (x y : ℕ), ¬ ((x, y) ∈ first_layer ∧ (x, y) ∈ second_layer)) ∧ 
    (∀ (x y : ℕ), (x, y) ∈ first_layer → (x, y) ∈ first_layer) ∧
    (∀ (x, y : ℕ), (x, y + 1) ∈ first_layer ∨ (x + 1, y) ∈ first_layer) ∧
    (∀ (x, y : ℕ), (x, y + 1) ∈ second_layer ∨ (x + 1, y) ∈ second_layer) := by
  sorry

end rectangle_covering_with_dominoes_l384_384903


namespace ellipse_minor_to_major_ratio_l384_384093

theorem ellipse_minor_to_major_ratio (Γ : Ellipse) (F₁ F₂ : Point)
  (P Q : Point) (h1 : Γ.foci = (F₁, F₂)) (h2 : Line_through F₁ intersects_at Γ P Q)
  (h3 : dist P F₂ = dist F₁ F₂) (h4 : 3 * dist P F₁ = 4 * dist Q F₁) :
  Γ.minor_axis / Γ.major_axis = 2 * Real.sqrt 6 / 7 :=
sorry

end ellipse_minor_to_major_ratio_l384_384093


namespace digits_solution_l384_384628

def solve_digits_problem (A B C D E F : ℕ) : Prop :=
  (A + D + E = 15) ∧
  (A + C + F = 15) ∧
  (B + D + F = 15) ∧
  (B + C + E = 15) ∧
  (A + B + F = 15) ∧
  (A + B + C = 15) ∧
  (D + E + F = 15) ∧
  ({A, B, C, D, E, F} = {1, 2, 3, 4, 5, 6})

theorem digits_solution : solve_digits_problem 4 1 2 5 6 3 :=
by
  -- Each condition explicitly listed
  show (4 + 5 + 6 = 15) ∧
       (4 + 2 + 3 = 15) ∧
       (1 + 5 + 3 = 15) ∧
       (1 + 2 + 6 = 15) ∧
       (4 + 1 + 3 = 15) ∧
       (4 + 1 + 2 = 15) ∧
       (5 + 6 + 3 = 15) ∧
       ({4, 1, 2, 5, 6, 3} = {1, 2, 3, 4, 5, 6}),
  -- Each arithmetic verification
  repeat {split}; norm_num,
  exact finset.ext (by simp),

end digits_solution_l384_384628


namespace oysters_eaten_l384_384200

-- Define the conditions in Lean
def Squido_oysters : ℕ := 200
def Crabby_oysters (Squido_oysters : ℕ) : ℕ := 2 * Squido_oysters

-- Statement to prove
theorem oysters_eaten (Squido_oysters Crabby_oysters : ℕ) (h1 : Crabby_oysters = 2 * Squido_oysters) : 
  Squido_oysters + Crabby_oysters = 600 :=
by
  sorry

end oysters_eaten_l384_384200


namespace calculate_h_odot_h_odot_h_l384_384307

def odot (x y : ℝ) : ℝ := x^2 + x * y - y^2

theorem calculate_h_odot_h_odot_h : 
  let h := 2 in h \odot (h \odot h) = -4 := by
  let h : ℝ := 2
  have h_h : h \odot h = h^2 + h * h - h^2 := by sorry  -- calculation for h \odot h
  have h_res : h \odot h = 4 := by sorry  -- simplified result for h \odot h
  show h \odot (h \odot h) = -4 from sorry

end calculate_h_odot_h_odot_h_l384_384307


namespace transform_C1_to_C2_l384_384793

-- Defining the curves C1 and C2
def C1 (x : ℝ) : ℝ := Real.sin x
def C2 (x : ℝ) : ℝ := Real.sin (1/2 * x - π / 3)

-- Definitions to illustrate the transformations
def stretch (f : ℝ → ℝ) (k : ℝ) (x : ℝ) : ℝ := f (k * x)
def shift (f : ℝ → ℝ) (h : ℝ) (x : ℝ) : ℝ := f (x - h)

-- Equivalent transformation problem
theorem transform_C1_to_C2 :
  C2 = shift (stretch C1 (1/2)) (2 * π / 3) :=
by
  -- The proof goes here
  sorry

end transform_C1_to_C2_l384_384793


namespace sum_of_integers_within_absolute_range_eq_zero_l384_384532

def is_within_absolute_range (n : ℤ) : Prop :=
  (Int.abs n > 2 ∧ Int.abs n < 7)

theorem sum_of_integers_within_absolute_range_eq_zero :
  (∑ i in (Finset.filter is_within_absolute_range (Finset.range 14).image (λ x, x - 7)), i) = 0 :=
by
  sorry

end sum_of_integers_within_absolute_range_eq_zero_l384_384532


namespace quadratic_form_decomposition_l384_384177

theorem quadratic_form_decomposition (a b c : ℝ) (h : ∀ x : ℝ, 8 * x^2 + 64 * x + 512 = a * (x + b) ^ 2 + c) :
  a + b + c = 396 := 
sorry

end quadratic_form_decomposition_l384_384177


namespace area_of_triangle_is_rational_l384_384550

theorem area_of_triangle_is_rational
  (x1 x2 x3 y1 y2 y3 : ℤ) :
  ∃ r : ℚ, r = 1/2 * abs ((x1 + 0.5) * (y2 - y3) + (x2 + 0.5) * (y3 - y1) + (x3 + 0.5) * (y1 - y2)) := 
by 
  sorry

end area_of_triangle_is_rational_l384_384550


namespace find_phi_l384_384603

theorem find_phi :
  ∀ φ : ℝ, 0 < φ ∧ φ < 90 → 
    (∃θ : ℝ, θ = 144 ∧ θ = 2 * φ ∧ (144 - θ) = 72) → φ = 81 :=
by
  intros φ h1 h2
  sorry

end find_phi_l384_384603


namespace adjacent_block_permutations_l384_384842

-- Define the set of digits
def digits : List ℕ := [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

-- Define the block of digits that must be adjacent
def block : List ℕ := [2, 5, 8]

-- Function to calculate permutations of a list (size n)
def fact (n : ℕ) : ℕ := Nat.factorial n

-- Calculate the total number of arrangements
def total_arrangements : ℕ := fact 8 * fact 3

-- The main theorem statement to be proved
theorem adjacent_block_permutations :
  total_arrangements = 241920 :=
by
  sorry

end adjacent_block_permutations_l384_384842


namespace right_angled_triangle_solution_l384_384242

-- Define the necessary constants
def t : ℝ := 504 -- area in cm^2
def c : ℝ := 65 -- hypotenuse in cm

-- The definitions of the right-angled triangle's properties
def is_right_angled_triangle (a b : ℝ) : Prop :=
  a ^ 2 + b ^ 2 = c ^ 2 ∧ a * b = 2 * t

-- The proof problem statement
theorem right_angled_triangle_solution :
  ∃ (a b : ℝ), is_right_angled_triangle a b ∧ ((a = 63 ∧ b = 16) ∨ (a = 16 ∧ b = 63)) :=
sorry

end right_angled_triangle_solution_l384_384242


namespace log_a_2_value_l384_384790

theorem log_a_2_value (a : ℝ) (h1 : (1/2)^a = (Real.sqrt 2)/2) : Real.logBase a 2 = -1 := 
by
  have a_eq : a = 1/2 := sorry
  rw [a_eq]
  rw [←Real.logBase_neg_inv a 2, neg_neg]
  exact sorry

end log_a_2_value_l384_384790


namespace prove_prob_scoring_200_points_prove_prob_scoring_at_least_300_points_l384_384611

noncomputable def prob_scoring_200_points (P_A1 P_A2 P_A3 : ℝ) : ℝ :=
  P_A1 * P_A2 * (1 - P_A3) + (1 - P_A1) * P_A2 * P_A3

noncomputable def prob_scoring_at_least_300_points (P_A1 P_A2 P_A3 : ℝ) : ℝ :=
  P_A1 * (1 - P_A2) * P_A3 + (1 - P_A1) * P_A2 * P_A3 + P_A1 * P_A2 * P_A3

theorem prove_prob_scoring_200_points :
  prob_scoring_200_points 0.8 0.7 0.6 = 0.26 := sorry
  
theorem prove_prob_scoring_at_least_300_points :
  prob_scoring_at_least_300_points 0.8 0.7 0.6 = 0.564 := sorry

end prove_prob_scoring_200_points_prove_prob_scoring_at_least_300_points_l384_384611


namespace circle_sine_intersections_l384_384846

noncomputable def circle_sine_intersection (r h : ℝ) : set ℝ :=
{x | x^2 + (sin x - h)^2 = r^2}

theorem circle_sine_intersections (r h : ℝ) :
  ∃ n : ℕ, ∃ S : finset ℝ, S.card = n ∧ ∀ x ∈ S, x^2 + (sin x - h)^2 = r^2 ∨ n = 0 :=
sorry

end circle_sine_intersections_l384_384846


namespace bottle_caps_per_person_l384_384868

noncomputable def initial_caps : Nat := 150
noncomputable def rebecca_caps : Nat := 42
noncomputable def alex_caps : Nat := 2 * rebecca_caps
noncomputable def total_caps : Nat := initial_caps + rebecca_caps + alex_caps
noncomputable def number_of_people : Nat := 6

theorem bottle_caps_per_person : total_caps / number_of_people = 46 := by
  sorry

end bottle_caps_per_person_l384_384868


namespace find_m_l384_384768

theorem find_m (m : ℝ) (h : arctan (1 / 2) = arctan ( (4 - m) / (m + 2))) : m = 2 := by
  sorry

end find_m_l384_384768


namespace symmetric_point_l384_384940

theorem symmetric_point (a b : ℝ) : 
  symmetric_to_point (a, b) (fun p : ℝ × ℝ => p.1 + p.2 = 0) = (-a, -b) :=
sorry

end symmetric_point_l384_384940


namespace decreases_as_x_increases_l384_384619

theorem decreases_as_x_increases (f : ℝ → ℝ) (h₁ : f = λ x, x^2 + 1) (h₂ : f = λ x, -x^2 + 1) 
  (h₃ : f = λ x, 2x + 1) (h₄ : f = λ x, -2x + 1) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ > f x₂) ↔ f = λ x, -2x + 1 :=
sorry

end decreases_as_x_increases_l384_384619


namespace probability_is_five_sevenths_l384_384915

noncomputable def probability_prime_or_odd : ℚ :=
  let balls := {1, 2, 3, 4, 5, 6, 7}
  let primes := {2, 3, 5, 7}
  let odds := {1, 3, 5, 7}
  let combined := primes ∪ odds
  let favorable_outcomes := combined.card
  let total_outcomes := balls.card
  favorable_outcomes / total_outcomes

theorem probability_is_five_sevenths :
  probability_prime_or_odd = 5 / 7 := 
sorry

end probability_is_five_sevenths_l384_384915


namespace terminating_decimal_count_number_of_terminating_decimals_l384_384668

theorem terminating_decimal_count {n : ℕ} (h1 : 1 ≤ n) (h2 : n ≤ 180) :
  (∃ k, n = 9 * k ∧ 1 ≤ 9 * k ∧ 9 * k ≤ 180) ↔ n ∈ (finset.range 21).image (λ k, 9 * k) :=
by sorry

theorem number_of_terminating_decimals :
  (finset.range 21).card = 20 :=
by sorry

end terminating_decimal_count_number_of_terminating_decimals_l384_384668


namespace quadratic_radicals_x_le_10_l384_384427

theorem quadratic_radicals_x_le_10 (a x : ℝ) (h1 : 3 * a - 8 = 17 - 2 * a) (h2 : 4 * a - 2 * x ≥ 0) : x ≤ 10 :=
by
  sorry

end quadratic_radicals_x_le_10_l384_384427


namespace least_possible_integer_of_friend_discussion_l384_384266

theorem least_possible_integer_of_friend_discussion (M : ℕ) :
  (∃ incorrect_positions : ℕ × ℕ, incorrect_positions.1 + 1 = incorrect_positions.2 ∧
    incorrect_positions.2 <= 20 ∧
    let correct_positions := {n | n ∈ finset.range 21 ∧ n ≠ incorrect_positions.1 ∧ n ≠ incorrect_positions.2}
    in ∀ n ∈ correct_positions, n > 0 → M % n = 0) →
  (M >= 1) → 
  M = 12252240 :=
begin
  sorry
end

end least_possible_integer_of_friend_discussion_l384_384266


namespace summer_school_friendship_l384_384438

theorem summer_school_friendship (n : ℕ) (h1 : 4 < n)
  (h2 : ∀ (a b : ℕ), a ≠ b → (∃ (f : finset ℕ), f.card = n ∧ 
  (∀ x y ∈ f, x ≠ y → (¬ a ∈ x ∨ ¬ b ∈ y)) ∧ (∀ x y ∈ f, x ≠ y → (a ∈ x ∧ b ∈ y → x ∩ y = ∅)) ))
  (h3 : ∀ (a b : ℕ), a ≠ b → ((∀ x y : finset ℕ, x ≠ y → x ∩ y = ∅) ∧ 
  (∀ x y : finset ℕ, x ≠ y → x ∩ y ≠ ∅ → x ≠ a ∧ x ≠ b → ∃ c : ℕ, c ∈ x ∧ c ∈ y ∧ c ≠ a ∧ c ≠ b))) :
  ∃ (k : ℕ), 8 * n - 7 = k^2 ∧ ∀ m, 4 < m → (∃ k, 8 * m - 7 = k^2) → n ≤ m := 
sorry

end summer_school_friendship_l384_384438


namespace mason_ate_15_hotdogs_l384_384893

structure EatingContest where
  hotdogWeight : ℕ
  burgerWeight : ℕ
  pieWeight : ℕ
  noahBurgers : ℕ
  jacobPiesLess : ℕ
  masonHotdogsWeight : ℕ

theorem mason_ate_15_hotdogs (data : EatingContest)
    (h1 : data.hotdogWeight = 2)
    (h2 : data.burgerWeight = 5)
    (h3 : data.pieWeight = 10)
    (h4 : data.noahBurgers = 8)
    (h5 : data.jacobPiesLess = 3)
    (h6 : data.masonHotdogsWeight = 30) :
    (data.masonHotdogsWeight / data.hotdogWeight) = 15 :=
by
  sorry

end mason_ate_15_hotdogs_l384_384893


namespace num_terminating_decimals_l384_384682

theorem num_terminating_decimals (n : ℕ) (h1 : 1 ≤ n ∧ n ≤ 180) :
  (∃ k : ℕ, k = 20) := by
  have h2 : ∀ n, (1 ≤ n ∧ n ≤ 180) → (∃ k, k = (180 / 9)) := sorry
  have h3 : 180 / 9 = 20 := by norm_num
  use 20
  exact h3

end num_terminating_decimals_l384_384682


namespace rooks_nonattacking_placements_l384_384403

def rook_placements : ℕ := 608

theorem rooks_nonattacking_placements :
  ∃ (w_positions b_positions : set (fin 4 × fin 4)),
    w_positions.card = 3 ∧
    b_positions.card = 3 ∧
    (∀ w ∈ w_positions, ∀ b ∈ b_positions, w.1 ≠ b.1 ∧ w.2 ≠ b.2) ∧
    (nat.card {pw | pw.fst ∈ w_positions ∧ pw.snd ∈ b_positions}) = rook_placements :=
sorry

end rooks_nonattacking_placements_l384_384403


namespace attendance_changes_l384_384612

theorem attendance_changes :
  let m := 25  -- Monday attendance
  let t := 31  -- Tuesday attendance
  let w := 20  -- initial Wednesday attendance
  let th := 28  -- Thursday attendance
  let f := 22  -- Friday attendance
  let sa := 26  -- Saturday attendance
  let w_new := 30  -- corrected Wednesday attendance
  let initial_total := m + t + w + th + f + sa
  let new_total := m + t + w_new + th + f + sa
  let initial_mean := initial_total / 6
  let new_mean := new_total / 6
  let mean_increase := new_mean - initial_mean
  let initial_median := (25 + 26) / 2  -- median of [20, 22, 25, 26, 28, 31]
  let new_median := (26 + 28) / 2  -- median of [22, 25, 26, 28, 30, 31]
  let median_increase := new_median - initial_median
  mean_increase = 1.667 ∧ median_increase = 1.5 := by
sorry

end attendance_changes_l384_384612


namespace function_expression_and_monotone_l384_384391

-- Given conditions
variables (A ω φ B : ℝ)
variable (k : ℤ)
variable (x : ℝ)
-- Given and variable ranges
variable (A_pos : A > 0)
variable (ω_pos : ω > 0)
variable (φ_bound : |φ| < π / 2)
variable (max_val : f(x) = A * sin(ω * x + φ) + B = 2 * sqrt 2)
variable (min_val : f(x) = A * sin(ω * x + φ) + B = - sqrt 2)
variable (periodic : f(x) = A * sin(ω * x + φ) + B)
variable (period : 2 * π / ω = π)
variable (passing_point : f(0) = - sqrt 2 / 4)

-- Monotonically increasing definition
def monotonically_increasing (f : ℝ → ℝ) : Prop :=
  ∀ a b, a < b → f(a) ≤ f(b)

noncomputable def f : ℝ → ℝ :=
  λ x, (3 * sqrt 2 / 2) * sin (2 * x - π / 6) + (sqrt 2 / 2)

theorem function_expression_and_monotone :
  (∀ x, f(x) = (3 * sqrt 2 / 2) * sin (2 * x - π / 6) + (sqrt 2 / 2)) ∧ 
  (∀ k : ℤ, monotonically_increasing f [k * π - π / 6, k * π + π / 3]) :=
sorry

end function_expression_and_monotone_l384_384391


namespace problem_solution_l384_384826

open Complex

theorem problem_solution :
  let B := 5 - 2 * I
  let N := -5 + 2 * I
  let T := 0 + 3 * I
  let Q := 3 + 0 * I
  B - N + T - Q = 7 - I :=
by
  let B := 5 - 2 * I
  let N := -5 + 2 * I
  let T := 0 + 3 * I
  let Q := 3 + 0 * I
  calc
    B - N + T - Q = (5 - 2 * I) - (-5 + 2 * I) + (0 + 3 * I) - (3 + 0 * I) : by rw [B, N, T, Q]
               ... = 7 - I                                          : sorry

end problem_solution_l384_384826


namespace linda_average_speed_l384_384486

def distance1 : ℝ := 420
def time1 : ℝ := 7.5
def distance2 : ℝ := 480
def time2 : ℝ := 8.25

def total_distance : ℝ := distance1 + distance2
def total_time : ℝ := time1 + time2
def average_speed : ℝ := total_distance / total_time

theorem linda_average_speed :
  average_speed = 57.14 :=
by
  -- proof to be filled (using sorry for now)
  sorry

end linda_average_speed_l384_384486


namespace line_through_parabola_and_midpoint_l384_384594

theorem line_through_parabola_and_midpoint 
  (focus_line : ∀ (A B : ℝ × ℝ), (A.1, A.2) = (1, 0) ∨ (B.1, B.2) = (1, 0) → ∃ (k : ℝ), (λ x, k * x))
  (intersection_parabola : ∀ (A B : ℝ × ℝ), A.2 ^ 2 = 4 * A.1 ∧ B.2 ^ 2 = 4 * B.1)
  (midpoint : ∀ (A B : ℝ × ℝ), ((A.1 + B.1) / 2, (A.2 + B.2) / 2) = (2, 1)) :
  ∃ (m b : ℝ), (λ x, m * x + b) = (λ x, x - 1) :=
sorry

end line_through_parabola_and_midpoint_l384_384594


namespace terminating_decimal_integers_count_l384_384739

/-- For how many integer values of n between 1 and 180 inclusive does the decimal 
representation of n / 180 terminate? We need to calculate the number of integers n such 
that n includes at least 9 as a factor. The answer is 20. -/
theorem terminating_decimal_integers_count :
  { n : ℤ | 1 ≤ n ∧ n ≤ 180 ∧ 9 ∣ n }.to_finset.card = 20 := 
by sorry

end terminating_decimal_integers_count_l384_384739


namespace maximize_profit_l384_384278

noncomputable def f (x : ℝ) : ℝ := x + 1

noncomputable def g : ℝ → ℝ := λ x,
  if 0 ≤ x ∧ x ≤ 3 then (10 * x + 1) / (x + 1)
  else if 3 < x ∧ x ≤ 5 then -x^2 + 9 * x - 12
  else 0

noncomputable def S (x : ℝ) : ℝ :=
  f (5 - x) + g x

theorem maximize_profit :
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ 5 ∧ S x = 11 :=
begin
  use 2,
  split,
  { norm_num },
  split,
  { norm_num },
  { sorry } -- Prove that S(2) = 11.
end

end maximize_profit_l384_384278


namespace part1_part2_l384_384808

open Real

-- Define the quadratic equation
def quadratic_eq (m x : ℝ) : ℝ := x^2 + (m + 2) * x + m

-- Part 1: Prove that the equation always has two distinct real roots
theorem part1 (m : ℝ) : discrim_quad (quadratic_eq m 0) > 0 := by sorry

-- Part 2: Given the equation has two real roots and the additional condition, find the value of m
theorem part2 (x1 x2 m : ℝ) 
  (h₀ : quadratic_eq m x1 = 0) 
  (h₁ : quadratic_eq m x2 = 0)
  (h₂ : x1 + x2 + 2 * x1 * x2 = 1) 
  : m = 3 := by sorry

end part1_part2_l384_384808


namespace solve_ineq_for_a_eq_0_values_of_a_l384_384804

theorem solve_ineq_for_a_eq_0 :
  ∀ x : ℝ, (|x + 2| - 3 * |x|) ≥ 0 ↔ (-1/2 <= x ∧ x <= 1) := 
by
  sorry

theorem values_of_a :
  ∀ x a : ℝ, (|x + 2| - 3 * |x|) ≥ a → (a ≤ 2) := 
by
  sorry

end solve_ineq_for_a_eq_0_values_of_a_l384_384804


namespace function_passes_through_fixed_point_l384_384988

theorem function_passes_through_fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x, f x = a^(x + 2) + 5) → f (-2) = 6 :=
by
  intro f_def
  have step1 := f_def (-2)
  rw [show (-2 + 2 = 0), by norm_num] at step1
  rw [show (a^0 = 1), by simp] at step1
  exact step1.symm -- This shows that f(-2) = 6

end function_passes_through_fixed_point_l384_384988


namespace angle_B_value_l384_384081

theorem angle_B_value (a b c A B : ℝ) (h1 : Real.sqrt 3 * a = 2 * b * Real.sin A) : 
  Real.sin B = Real.sqrt 3 / 2 ↔ (B = Real.pi / 3 ∨ B = 2 * Real.pi / 3) :=
by sorry

noncomputable def find_b_value (a : ℝ) (area : ℝ) (A B c : ℝ) (h1 : a = 6) (h2 : area = 6 * Real.sqrt 3) (h3 : c = 4) (h4 : B = Real.pi / 3 ∨ B = 2 * Real.pi / 3) : 
  ℝ := 
if B = Real.pi / 3 then 2 * Real.sqrt 7 else Real.sqrt 76

end angle_B_value_l384_384081


namespace boiling_point_fahrenheit_l384_384560

theorem boiling_point_fahrenheit (boiling_point_celsius : ℕ) (boiling_point_fahrenheit : ℕ) 
  (melting_point_celsius : ℕ) (melting_point_fahrenheit : ℕ)
  (temp_celsius : ℕ) (temp_fahrenheit : ℕ) :
  boiling_point_celsius = 100 → melting_point_celsius = 0 → 
  boiling_point_fahrenheit = 212 → melting_point_fahrenheit = 32 → 
  temp_celsius = 35 → temp_fahrenheit = 95 →
  boiling_point_fahrenheit = 212 := by
  intros
  exact hboiling_point_fahrenheit

end boiling_point_fahrenheit_l384_384560


namespace largest_divisor_l384_384420

theorem largest_divisor (n : ℕ) (hn : n > 0) (h : 360 ∣ n^3) :
  ∃ w : ℕ, w > 0 ∧ w ∣ n ∧ ∀ d : ℕ, (d > 0 ∧ d ∣ n) → d ≤ 30 := 
sorry

end largest_divisor_l384_384420


namespace cos_shifted_alpha_l384_384761

theorem cos_shifted_alpha (α : ℝ) (h1 : Real.tan α = -3/4) (h2 : α ∈ Set.Ioc (3*Real.pi/2) (2*Real.pi)) :
  Real.cos (Real.pi/2 + α) = 3/5 :=
sorry

end cos_shifted_alpha_l384_384761


namespace terminating_decimal_fraction_count_l384_384689

theorem terminating_decimal_fraction_count :
  (finset.filter (λ n : ℕ, 1 ≤ n ∧ n ≤ 180 ∧ 90 ∣ n) (finset.range 181)).card = 20 :=
by sorry

end terminating_decimal_fraction_count_l384_384689


namespace probability_of_valid_triangle_in_15gon_l384_384548

def regular_15gon_segment_length (k : ℕ) : ℝ :=
  2 * Real.sin(k * Real.pi / 15)

def total_segments : ℕ :=
  Nat.choose 15 2

def valid_triangle_combinations : ℕ := 
  -- Placeholder for actual count of valid combinations
  -- to be filled in with correct calculation.
  sorry 

def total_combinations : ℕ :=
  Nat.choose total_segments 3

def probability_valid_triangle : ℚ :=
  valid_triangle_combinations / total_combinations

theorem probability_of_valid_triangle_in_15gon :
  probability_valid_triangle = 323 / 429 :=
by
  sorry

end probability_of_valid_triangle_in_15gon_l384_384548


namespace sequence_2019_value_l384_384962

theorem sequence_2019_value :
  ∃ a : ℕ → ℤ, (∀ n ≥ 4, a n = a (n-1) * a (n-3)) ∧ a 1 = 1 ∧ a 2 = 1 ∧ a 3 = -1 ∧ a 2019 = -1 :=
by
  sorry

end sequence_2019_value_l384_384962


namespace p_divisible_by_1979_l384_384824

theorem p_divisible_by_1979 (p q : ℕ) 
  (h1 : (p : ℚ) / q = ∑ i in Finset.range 1320, (-1) ^ i * (1 / (i + 1))) : 
  1979 ∣ p :=
sorry

end p_divisible_by_1979_l384_384824


namespace terminating_decimal_fraction_count_l384_384686

theorem terminating_decimal_fraction_count :
  (finset.filter (λ n : ℕ, 1 ≤ n ∧ n ≤ 180 ∧ 90 ∣ n) (finset.range 181)).card = 20 :=
by sorry

end terminating_decimal_fraction_count_l384_384686


namespace factory_processing_capacity_l384_384843

theorem factory_processing_capacity :
  ∃ (x y : ℕ), (1200 / x = 1200 / y + 10) ∧ (y = 3 * x / 2) ∧ (x = 40) ∧ (y = 60) :=
by {
  let x := 40,
  let y := 60,
  use [x, y],
  split,
  sorry, -- Proof of the first condition
  split,
  exact (by norm_num : y = 3 * x / 2),
  split,
  exact (by norm_num : x = 40),
  exact (by norm_num : y = 60)
}

end factory_processing_capacity_l384_384843


namespace compute_Barium_atoms_l384_384257

-- Define the atomic weights of Barium and Fluorine
def atomic_weight_Barium : ℝ := 137.33
def atomic_weight_Fluorine : ℝ := 18.998

-- Define the number of Fluorine atoms and molecular weight of the compound
def number_of_Fluorine_atoms : ℕ := 2
def molecular_weight_compound : ℝ := 175

-- The expected number of Barium atoms in the compound
def number_of_Barium_atoms : ℕ := 1

-- Prove that the number of Barium atoms in the compound is 1
theorem compute_Barium_atoms :
  let weight_Fluorine := number_of_Fluorine_atoms * atomic_weight_Fluorine,
      weight_Barium := molecular_weight_compound - weight_Fluorine,
      computed_Barium_atoms := weight_Barium / atomic_weight_Barium in
  rounded(computed_Barium_atoms) = number_of_Barium_atoms := sorry

end compute_Barium_atoms_l384_384257


namespace trapezium_other_side_length_l384_384209

theorem trapezium_other_side_length (a h Area : ℕ) (a_eq : a = 4) (h_eq : h = 6) (Area_eq : Area = 27) : 
  ∃ (b : ℕ), b = 5 := 
by
  sorry

end trapezium_other_side_length_l384_384209


namespace range_of_even_function_l384_384408

noncomputable def f (a b x : ℝ) : ℝ := a * x ^ 2 + b * x + 2

theorem range_of_even_function (a b : ℝ) (h1 : ∀ x, f a b x = f a b (-x))
  (h2 : 1 + a = -2) :
  set.range (λ x, f a b x) = {y : ℝ | y ≤ 2 ∧ y ≥ -10} :=
sorry

end range_of_even_function_l384_384408


namespace largest_possible_integer_in_list_l384_384595

theorem largest_possible_integer_in_list :
  ∃ (a b c d e : ℕ), 
    (a = 7) ∧ (b = 7) ∧ (c = 10) ∧ (d > 10) ∧ (a + b + c + d + e = 60) 
    ∧ (List.median [a, b, c, d, e] = 10) 
    ∧ (List.repeated [a, b, c, d, e] = [7])
    → e = 25 :=
by
  sorry

end largest_possible_integer_in_list_l384_384595


namespace tangent_line_equation_at_point_l384_384161

theorem tangent_line_equation_at_point :
  let y := λ x : ℝ, (1 + x) / (1 - x)
  let x₀ := 2
  let y₀ := -3
  let m := 2
  (2 * x₀ - y₀ - 7 = 0) :=
by 
  sorry

end tangent_line_equation_at_point_l384_384161


namespace sum_distinct_possible_values_of_GH_is_225_l384_384172

/-- Given a digit H which is part of a number 541G5072H6, 
    and the product GH where G ranges from 0 to 9,
    if the number is divisible by 40, then the sum of all distinct 
    possible values of GH is 225. -/
theorem sum_distinct_possible_values_of_GH_is_225 (G H : ℕ) (HG_bounds : G ∈ finset.range 10) (H_eq_5 : H = 5) :
  (finset.range 10).sum (λ G, G * H) = 225 :=
by
  sorry

end sum_distinct_possible_values_of_GH_is_225_l384_384172


namespace reflected_line_equation_l384_384943

-- Definitions based on given conditions
def incident_line (x : ℝ) : ℝ := 2 * x + 1
def reflection_line (x : ℝ) : ℝ := x

-- Statement of the mathematical problem
theorem reflected_line_equation :
  ∀ x y : ℝ, (incident_line x = y) → (reflection_line x = x) → y = (1/2) * x - (1/2) :=
sorry

end reflected_line_equation_l384_384943


namespace alpha_plus_beta_l384_384147

noncomputable def alpha_beta_equation (α β : ℝ) : Prop :=
  α ∈ Set.Ioc 0 (π / 2) ∧ β ∈ Set.Ioc 0 (π / 2) ∧ 
  Real.tan β = (Real.cot α - 1) / (Real.cot α + 1)

theorem alpha_plus_beta (α β : ℝ) (h : alpha_beta_equation α β) : α + β = π / 4 :=
  sorry

end alpha_plus_beta_l384_384147


namespace smallest_possible_value_of_product_sum_l384_384472

noncomputable def g (x : ℝ) : ℝ := x^4 + 18 * x^3 + 98 * x^2 + 108 * x + 36 

theorem smallest_possible_value_of_product_sum : 
  ∀ w1 w2 w3 w4 : ℝ, (roots g = [w1, w2, w3, w4]) → 
  (∀ a b c d : {x : ℕ // 1 ≤ x ∧ x ≤ 4}, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d → 
  |w1 * w2 + w3 * w4| ≥ 24) ∧ (∃ a b c d : {x : ℕ // 1 ≤ x ∧ x ≤ 4}, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  |w1 * w2 + w3 * w4| = 24) :=
by
  sorry

end smallest_possible_value_of_product_sum_l384_384472


namespace probability_chocolate_milk_5_out_of_7_l384_384129

theorem probability_chocolate_milk_5_out_of_7 :
  let p := (3 / 4 : ℚ) in
  let n := 7 in
  let k := 5 in
  (nat.choose n k * p^k * (1 - p)^(n - k)) = (5103 / 16384 : ℚ) := sorry

end probability_chocolate_milk_5_out_of_7_l384_384129


namespace terminating_decimals_l384_384699

theorem terminating_decimals (n : ℤ) (h : 1 ≤ n ∧ n ≤ 180) :
  (∃ k : ℤ, k = 20 ∧ ∀ n, 1 ≤ n ∧ n ≤ 180 → (∃ m, m * 180 = n * (2^2 * 5))) := by
  sorry

end terminating_decimals_l384_384699


namespace is_center_on_circumcircle_l384_384941

open EuclideanGeometry

structure IsoscelesTrapezoid (A B C D : Point) : Prop :=
(is_isosceles : ∀{M : Point}, is_median M A B D C → A = M ∨ B = M ∨ C = M ∨ D = M)

theorem is_center_on_circumcircle {A B C D P O : Point}
  (h_trapezoid : IsoscelesTrapezoid A B C D)
  (h_intersect : Line A C ∩ Line B D = {P})
  (h_circumcenter : IsCircumcenter O A B C D) :
  O ∈ Circumcircle A B P := 
sorry

end is_center_on_circumcircle_l384_384941


namespace h_eq_20_imp_x_eq_30_div_7_l384_384931

noncomputable def h (x : ℝ) : ℝ := 4 * (f x)
noncomputable def f (x : ℝ) : ℝ := 30 / (x + 2)
noncomputable def f_inv (x : ℝ) : ℝ := 1 / (f x)

theorem h_eq_20_imp_x_eq_30_div_7 (x : ℝ) :
  h x = 20 → x = 30 / 7 := 
sorry

end h_eq_20_imp_x_eq_30_div_7_l384_384931


namespace train_speed_is_180_kmh_l384_384252

-- Defining the given conditions
def train_length_m : ℝ := 150 -- Train length in meters
def crossing_time_s : ℝ := 3 -- Time to cross in seconds

-- Conversion factors
def meter_to_kilometer : ℝ := 1 / 1000 -- Conversion factor from meters to kilometers
def second_to_hour : ℝ := 1 / 3600 -- Conversion factor from seconds to hours

-- Defining the calculation for speed in km/h
noncomputable def speed_kmh : ℝ := 
  (train_length_m * meter_to_kilometer) / (crossing_time_s * second_to_hour)

-- Theorem to prove the speed of the train
theorem train_speed_is_180_kmh : speed_kmh = 180 := by
  sorry

end train_speed_is_180_kmh_l384_384252


namespace bead_problem_l384_384860

theorem bead_problem 
  (x y : ℕ) 
  (hx : 19 * x + 17 * y = 2017): 
  (x + y = 107) ∨ (x + y = 109) ∨ (x + y = 111) ∨ (x + y = 113) ∨ (x + y = 115) ∨ (x + y = 117) := 
sorry

end bead_problem_l384_384860


namespace domain_f_2x_l384_384053

def domain_f := {x : ℝ | 1 ≤ x ∧ x ≤ 4}

theorem domain_f_2x : 
  (∀ f : ℝ → ℝ, ∀ x, x ∈ {x | ∃ y ∈ domain_f, f(y) = f(2^x)}) ↔ (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2) :=
by
  sorry

end domain_f_2x_l384_384053


namespace sin_double_angle_l384_384498

noncomputable def points_equidistant (A B C D P : Type*) [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space P] := 
(dist A B) = (dist B C) ∧ (dist B C) = (dist C D)

def cos_angles (A P C B D : Type*) [metric_space A] [metric_space P] [metric_space C] [metric_space B] [metric_space D] :=
  ∃cos_APC: (cos_same_line A P C) = 3/5 ∧ (cos_same_line B P D) = 1/5

theorem sin_double_angle (A B C D P: Type*) [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space P]
  (h1 : points_equidistant A B C D P)
  (h2 : cos_angles A P C B D) :
  ∃θ : Type*, sin (2 * angle B P C) = 8 * sqrt 6 / 25 :=
by
  sorry

end sin_double_angle_l384_384498


namespace range_of_f_length_AD_l384_384390

open Real

-- Define the function f(x)
def f (x : ℝ) : ℝ := 4 * sin x * sin (x + π / 3)

-- Problem 1: Range of f(x) when x ∈ [0, π/2]
theorem range_of_f : ∀ x ∈ Icc 0 (π / 2), f(x) ∈ Icc 0 3 := sorry

-- Definitions for Problem 2
variables {A B C : ℝ} (b : ℝ) (c : ℝ) (D : ℝ)
-- Midpoint formula specific for this problem and simplifying the context for Lean
def midpoint_dist (b c A : ℝ) : ℝ := sqrt ((c^2 + b^2 + c * b) / 4)

-- Problem 2: Length of AD given the conditions
theorem length_AD (h1 : b = 2) (h2 : c = 4) (h3 : ∀ x : ℝ, f(x) ≤ f(A)) : 
  midpoint_dist _ _ _ = sqrt 7 := sorry

end range_of_f_length_AD_l384_384390


namespace find_center_of_circle_l384_384589

-- Condition 1: The circle is tangent to the lines 3x - 4y = 12 and 3x - 4y = -48
def tangent_line1 (x y : ℝ) : Prop := 3 * x - 4 * y = 12
def tangent_line2 (x y : ℝ) : Prop := 3 * x - 4 * y = -48

-- Condition 2: The center of the circle lies on the line x - 2y = 0
def center_line (x y : ℝ) : Prop := x - 2 * y = 0

-- The center of the circle
def circle_center (x y : ℝ) : Prop := 
  tangent_line1 x y ∧ tangent_line2 x y ∧ center_line x y

-- Statement to prove
theorem find_center_of_circle : 
  circle_center (-18) (-9) := 
sorry

end find_center_of_circle_l384_384589


namespace terminating_decimal_fraction_count_l384_384688

theorem terminating_decimal_fraction_count :
  (finset.filter (λ n : ℕ, 1 ≤ n ∧ n ≤ 180 ∧ 90 ∣ n) (finset.range 181)).card = 20 :=
by sorry

end terminating_decimal_fraction_count_l384_384688


namespace prob_inclination_angle_eqn_l384_384531

noncomputable def inclination_angle (a b c : ℝ) : ℝ :=
  real.arctan (-a / b)

theorem prob_inclination_angle_eqn :
  inclination_angle 1 (-√3) (-2014) = π / 6 :=
by
  sorry

end prob_inclination_angle_eqn_l384_384531


namespace terminating_decimal_integers_count_l384_384742

/-- For how many integer values of n between 1 and 180 inclusive does the decimal 
representation of n / 180 terminate? We need to calculate the number of integers n such 
that n includes at least 9 as a factor. The answer is 20. -/
theorem terminating_decimal_integers_count :
  { n : ℤ | 1 ≤ n ∧ n ≤ 180 ∧ 9 ∣ n }.to_finset.card = 20 := 
by sorry

end terminating_decimal_integers_count_l384_384742


namespace concyclic_points_l384_384845

theorem concyclic_points 
  (A B C D K N L M : Point)
  (square : Square A B C D)
  (on_AB : K ∈ Segment A B)
  (on_AD : N ∈ Segment A D)
  (intersect_l : L ∈ LineSegment (C, K ∩ Diagonal B D))
  (intersect_m : M ∈ LineSegment (C, N ∩ Diagonal B D))
  (condition : AK * AN = 2 * BK * DN) :
  Concyclic {K, L, M, N, A} :=
  sorry

end concyclic_points_l384_384845


namespace radius_of_convergence_1_correct_radius_of_convergence_2_correct_radius_of_convergence_3_correct_l384_384660

noncomputable def radius_of_convergence_1 : ℂ → ℝ :=
  sorry

theorem radius_of_convergence_1_correct (z : ℂ) :
  radius_of_convergence_1 (z) = 1 / 2 :=
  sorry

noncomputable def radius_of_convergence_2 : ℂ → ℝ :=
  sorry

theorem radius_of_convergence_2_correct (z : ℂ) :
  radius_of_convergence_2 (z) = ∞ :=
  sorry

noncomputable def radius_of_convergence_3 : ℂ → ℝ :=
  sorry

theorem radius_of_convergence_3_correct (z : ℂ) :
  radius_of_convergence_3 (z) = 1 / (Real.cbrt 5) :=
  sorry

end radius_of_convergence_1_correct_radius_of_convergence_2_correct_radius_of_convergence_3_correct_l384_384660


namespace triangle_side_relation_l384_384511

theorem triangle_side_relation (a b c : ℝ) 
    (h_angles : 55 = 55 ∧ 15 = 15 ∧ 110 = 110) :
    c^2 - a^2 = a * b :=
  sorry

end triangle_side_relation_l384_384511


namespace books_in_school_libraries_correct_l384_384195

noncomputable def booksInSchoolLibraries : ℕ :=
  let booksInPublicLibrary := 1986
  let totalBooks := 7092
  totalBooks - booksInPublicLibrary

-- Now we create a theorem to check the correctness of our definition
theorem books_in_school_libraries_correct :
  booksInSchoolLibraries = 5106 := by
  sorry -- We skip the proof, as instructed

end books_in_school_libraries_correct_l384_384195


namespace find_a_l384_384024

theorem find_a (f : ℝ → ℝ) (a : ℝ) 
  (h1 : ∀ (x : ℝ), f x = a^(x+1) - 2) 
  (h2 : a > 0 ∧ a ≠ 1)
  (h3 : f(-(-1)) = 2) : a = 2 :=
by
  sorry

end find_a_l384_384024


namespace colonization_combinations_l384_384822
open Nat

theorem colonization_combinations : 
  let earth_like := 6
  let mars_like := 7
  let total_resources := 15
  colonization (total_resources earth_like mars_like) = 336 :=
by
  sorry

end colonization_combinations_l384_384822


namespace max_towns_meeting_criteria_l384_384564

-- Define the graph with edges of types air, bus, and train
inductive Link
| air
| bus
| train

-- Define a structure for the town network
structure Network (V : Type*) :=
(edges : V → V → Option Link)
(pairwise_linked : ∀ u v : V, u ≠ v → ∃ (lk : Link), edges u v = some lk)
(has_air_link : ∃ u v : V, u ≠ v ∧ edges u v = some Link.air)
(has_bus_link : ∃ u v : V, u ≠ v ∧ edges u v = some Link.bus)
(has_train_link : ∃ u v : V, u ≠ v ∧ edges u v = some Link.train)
(no_all_three_types : ∀ v : V, ¬ (∃ u w: V, u ≠ v ∧ w ≠ v ∧ u ≠ w ∧
  edges v u = some Link.air ∧ edges v w = some Link.bus ∧ edges v w = some Link.train))
(no_same_type_triangle : ∀ u v w : V, u ≠ v ∧ v ≠ w ∧ u ≠ w →
  ¬ (edges u v = edges u w ∧ edges u w = edges v w))

-- The theorem stating the maximum number of towns meeting the criteria
theorem max_towns_meeting_criteria (V : Type*) [Fintype V] [DecidableEq V] (h : ∀ e : V, ∃ p : Network V, p.edges = e) : 
  Fintype.card V ≤ 4 :=
by sorry

end max_towns_meeting_criteria_l384_384564


namespace terminating_decimal_fraction_count_l384_384690

theorem terminating_decimal_fraction_count :
  (finset.filter (λ n : ℕ, 1 ≤ n ∧ n ≤ 180 ∧ 90 ∣ n) (finset.range 181)).card = 20 :=
by sorry

end terminating_decimal_fraction_count_l384_384690


namespace tank_emptying_time_l384_384269

theorem tank_emptying_time :
  ∀ (C : ℝ) (Ltime : ℝ) (Irate : ℝ) (Lrate : ℝ) (Itime : ℝ),
    C = 3600.000000000001 ∧ Ltime = 6 ∧ Irate = 2.5 ∧ Lrate = 600 ∧ Lrate = C / Ltime ∧ Irate = 150 ∧ Itime = C / (Lrate - Irate) → Itime = 8 :=
by
  intros C Ltime Irate Lrate Itime
  intro h
  cases h with hc h
  cases h with ht hl
  cases h with hr hi
  cases hi with hrate him
  simp only [hc, ht, hr, hl, hrate, him]
  sorry

end tank_emptying_time_l384_384269


namespace uncle_vasya_can_get_rich_l384_384849

def initial_dollars := 100
def target_dollars := 200
def coconut_to_dollar := (1/15 : ℚ)
def dollar_to_coconut := 10
def enot_to_banana := 6
def enot_to_coconut := 11
def banana_to_coconut := 2

lemma exchange_cycle_profits (n : ℕ) : 
  (initial_dollars * dollar_to_coconut + n * (banana_to_coconut - enot_to_coconut)) * coconut_to_dollar ≥ target_dollars :=
  sorry

theorem uncle_vasya_can_get_rich :
  ∃ (n : ℕ), (initial_dollars * dollar_to_coconut + n * (banana_to_coconut - enot_to_coconut)) * coconut_to_dollar ≥ target_dollars := 
  exists.intro 2000 (exchange_cycle_profits 2000)

end uncle_vasya_can_get_rich_l384_384849


namespace terminating_decimals_l384_384696

theorem terminating_decimals (n : ℤ) (h : 1 ≤ n ∧ n ≤ 180) :
  (∃ k : ℤ, k = 20 ∧ ∀ n, 1 ≤ n ∧ n ≤ 180 → (∃ m, m * 180 = n * (2^2 * 5))) := by
  sorry

end terminating_decimals_l384_384696


namespace angle_bfc_right_from_conditions_l384_384776

open_locale classical

noncomputable theory

variables (A B C B1 C1 P Q F : Type) [has_angle A B C] 

def acute_triangle (ABC : Type) := sorry
def is_height (B B1 : Type) (ABC : Type) := sorry
def points_extended_past (B1 C1 P Q : Type) := sorry
def angle_right (A P Q : Type) := sorry
def height_apq (AF PQ : Type) := sorry
def angle_bfc_right (B F C : Type) := sorry

theorem angle_bfc_right_from_conditions (ABC : Type) (B B1 C C1 P Q F : Type)
  [acute_triangle ABC]
  [is_height B B1 ABC]
  [is_height C C1 ABC]
  [points_extended_past B1 C1 P Q]
  [angle_right A P Q]
  [height_apq AF (triangle A P Q)]
  : angle_bfc_right B F C := sorry

end angle_bfc_right_from_conditions_l384_384776


namespace profit_without_discount_l384_384608

noncomputable def cost_price : ℝ := 100
noncomputable def discounted_selling_price : ℝ := 142.50
noncomputable def discount_rate : ℝ := 0.05
noncomputable def profit_with_discount : ℝ := 0.425

-- Target statement we want to prove
theorem profit_without_discount :
  let original_selling_price := discounted_selling_price / (1 - discount_rate) in 
  let profit_without_discount := original_selling_price - cost_price in
  let profit_percentage_without_discount := (profit_without_discount / cost_price) * 100 in
  profit_percentage_without_discount = 50 :=
by
  sorry

end profit_without_discount_l384_384608


namespace num_terminating_decimals_l384_384675

theorem num_terminating_decimals (n : ℕ) (h1 : 1 ≤ n ∧ n ≤ 180) :
  (∃ k : ℕ, k = 20) := by
  have h2 : ∀ n, (1 ≤ n ∧ n ≤ 180) → (∃ k, k = (180 / 9)) := sorry
  have h3 : 180 / 9 = 20 := by norm_num
  use 20
  exact h3

end num_terminating_decimals_l384_384675


namespace total_vowels_written_l384_384106

theorem total_vowels_written (n k : ℕ) (h1 : n = 5) (h2 : k = 3) : n * k = 15 := 
by 
  rw [h1, h2]
  norm_num

end total_vowels_written_l384_384106


namespace factorial_not_prime_l384_384308

theorem factorial_not_prime (n : ℕ) (h : n > 1) : ¬ nat.prime (nat.factorial n) :=
by {
  sorry
}

end factorial_not_prime_l384_384308


namespace max_value_at_one_l384_384022

noncomputable def f (x a b : ℝ) : ℝ := x^3 + a*x^2 + b*x - a^2 - 7*a

theorem max_value_at_one (a b : ℝ) (h1 : f 1 a b = 10) (h2 : ∀ x, (deriv (λ x, f x a b)) x = 3*x^2 + 2*a*x + b) (h3 : (deriv (λ x, f x a b)) 1 = 0) : a / b = -2 / 3 :=
by
  sorry

end max_value_at_one_l384_384022


namespace alpha_in_second_quadrant_l384_384045

theorem alpha_in_second_quadrant (α : ℝ) 
  (h1 : sin α * cos α < 0) 
  (h2 : cos α - sin α < 0) : 
  ∃ n : ℤ, π/2 + 2*n*π < α ∧ α < π + 2*n*π :=
by
  -- proof goes here
  sorry

end alpha_in_second_quadrant_l384_384045


namespace flour_already_put_in_l384_384892

-- Define the conditions
def total_flour_required := 12
def flour_to_be_added := 1

-- Define the statement to be proven
theorem flour_already_put_in : 
  (total_flour_required - flour_to_be_added) = 11 :=
by
  -- The proof is currently omitted; we're only setting up the structure here.
  Sorry

end flour_already_put_in_l384_384892


namespace flow_chart_output_l384_384651

theorem flow_chart_output :
  let S := 1 in
  let i := 1 in
  let final_S := Nat.iterate (fun (Si: Nat × Nat) => (Si.1 * 2, Si.2 + 1)) 4 (S, i) in
  final_S.1 = 16 := by
  sorry

end flow_chart_output_l384_384651


namespace f_reciprocal_sum_f_series_l384_384799

def f (x : ℝ) : ℝ := x^2 / (1 + x^2)

theorem f_reciprocal (x : ℝ) (h : x ≠ 0) : f x + f (1 / x) = 1 := 
by
  -- Proof will go here
  sorry

theorem sum_f_series : 
  ∑ k in (Finset.range 2017).map Nat.succ |> Finset.insert 1,
  f k + f (1 / k) = 2016.5 := 
by
  -- Proof will go here
  sorry

end f_reciprocal_sum_f_series_l384_384799


namespace number_of_possible_values_of_S_l384_384104

open Finset

theorem number_of_possible_values_of_S :
  let A := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
            21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 
            39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 
            57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 
            75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 
            93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 
            109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120 }
  in
  ∃ (S : ℕ), (∀ A ⊆ range 121, A.card = 80 → S = ∑ x in A, id x) ∧
             let S_min := ∑ i in range 1 (80+1), i in
             let S_max := ∑ i in range 41 (120+1), i in
             (max_value : ℕ) (min_value : ℕ), max_value = S_max ∧ min_value = S_min ∧
             max_value - min_value + 1 = 3201 :=
by
  sorry

end number_of_possible_values_of_S_l384_384104


namespace not_prime_sum_of_products_l384_384120

theorem not_prime_sum_of_products
  (a b c d : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  ¬ ∃ p, p.prime ∧ p = a * b + b * c + c * d + d * a :=
by
  sorry

end not_prime_sum_of_products_l384_384120


namespace conjugate_z_l384_384423

theorem conjugate_z (z : ℂ) (h : z * (1 + I) = -2 * I) : conj z = -1 + I :=
by
  sorry

end conjugate_z_l384_384423


namespace range_of_a_l384_384807

open Real

theorem range_of_a {a : ℝ} :
  (∀ x ∈ set.Icc (1 : ℝ) (2 : ℝ), 3 * x^2 - a ≥ 0) ∧
  (∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0) ↔
  (a ≤ -2 ∨ (1 ≤ a ∧ a ≤ 3)) := 
  sorry

end range_of_a_l384_384807


namespace triangle_angle_difference_triangle_area_l384_384082

theorem triangle_angle_difference (A B C a b c : ℝ) (h1 : b * sin B - a * sin A = c) : B - A = π / 2 :=
sorry

theorem triangle_area (A B C a b c : ℝ) (h1 : b * sin B - a * sin A = c) (h2 : c = √3) (h3 : C = π / 3) : 
  1 / 2 * a * c * sin B = √3 / 4 :=
sorry

end triangle_angle_difference_triangle_area_l384_384082


namespace hyperbola_eccentricity_range_l384_384027

theorem hyperbola_eccentricity_range (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h_intersect : ∃ x y : ℝ, (y = 2 * x) ∧ ((x^2 / a^2) - (y^2 / b^2) = 1)) : 
  (∃ e : ℝ, (e > sqrt 5) ∧ ∀ e', e' ≥ e) :=
sorry

end hyperbola_eccentricity_range_l384_384027


namespace union_sets_l384_384464

-- Define the sets A and B using their respective conditions.
def A : Set ℝ := {x : ℝ | 3 < x ∧ x ≤ 7}
def B : Set ℝ := {x : ℝ | 4 < x ∧ x ≤ 10}

-- The theorem we aim to prove.
theorem union_sets : A ∪ B = {x : ℝ | 3 < x ∧ x ≤ 10} := 
by
  sorry

end union_sets_l384_384464


namespace regular_tetrahedron_of_congruent_dihedral_angles_not_necessarily_regular_of_five_congruent_dihedral_angles_l384_384250

-- Part (a)
theorem regular_tetrahedron_of_congruent_dihedral_angles 
  {A B C D : Type} 
  (tetrahedron : tetrahedron A B C D)
  (congruent_dihedral_angles : ∀ P Q R S, P ≠ Q ∧ Q ≠ R ∧ R ≠ S ∧ S ≠ P → dihedral_angle (tetrahedron) P Q = dihedral_angle (tetrahedron) R S) :
  regular_tetrahedron tetrahedron :=
sorry

-- Part (b)
theorem not_necessarily_regular_of_five_congruent_dihedral_angles 
  {A B C D : Type} 
  (tetrahedron : tetrahedron A B C D)
  (five_congruent_dihedral_angles : ∃ P Q R S T,
      P ≠ Q ∧ Q ≠ R ∧ R ≠ S ∧ S ≠ T ∧ T ≠ P ∧ dihedral_angle (tetrahedron) P Q = dihedral_angle (tetrahedron) R S ∧ dihedral_angle (tetrahedron) P Q = dihedral_angle (tetrahedron) S T ∧ dihedral_angle (tetrahedron) P Q = dihedral_angle (tetrahedron) T P ∧ dihedral_angle (tetrahedron) P Q = dihedral_angle (tetrahedron) Q T) :
  ¬ regular_tetrahedron tetrahedron :=
sorry

end regular_tetrahedron_of_congruent_dihedral_angles_not_necessarily_regular_of_five_congruent_dihedral_angles_l384_384250


namespace terminating_decimals_l384_384703

theorem terminating_decimals (n : ℤ) (h : 1 ≤ n ∧ n ≤ 180) :
  (∃ k : ℤ, k = 20 ∧ ∀ n, 1 ≤ n ∧ n ≤ 180 → (∃ m, m * 180 = n * (2^2 * 5))) := by
  sorry

end terminating_decimals_l384_384703


namespace expression_value_as_fraction_l384_384316

theorem expression_value_as_fraction :
  2 + (3 / (2 + (5 / (4 + (7 / 3))))) = 91 / 19 :=
by
  sorry

end expression_value_as_fraction_l384_384316


namespace possible_ticket_prices_l384_384957

theorem possible_ticket_prices (x : ℕ) (hx : even x) (ticket_cost_class1 : x ∣ 72) (ticket_cost_class2 : x ∣ 108) : 
    finset.card ((finset.filter even (nat.divisors (nat.gcd 72 108)))) = 6 :=
by
  sorry

end possible_ticket_prices_l384_384957


namespace goldfish_at_surface_l384_384203

variable (total_goldfish : ℕ)
variable (below_surface_goldfish : ℕ)
variable (surface_fraction : ℚ)

theorem goldfish_at_surface (h1 : surface_fraction = 0.25)
                            (h2 : below_surface_goldfish = 45)
                            (h3 : below_surface_goldfish = total_goldfish * 0.75) :
  total_goldfish * surface_fraction = 15 := 
by
  sorry

end goldfish_at_surface_l384_384203


namespace length_of_train_is_400_meters_l384_384286

noncomputable def relative_speed (speed_train speed_man : ℝ) : ℝ :=
  speed_train - speed_man

noncomputable def km_per_hr_to_m_per_s (speed_km_per_hr : ℝ) : ℝ :=
  speed_km_per_hr * (1000 / 3600)

noncomputable def length_of_train (relative_speed_m_per_s time_seconds : ℝ) : ℝ :=
  relative_speed_m_per_s * time_seconds

theorem length_of_train_is_400_meters :
  let speed_train := 30 -- km/hr
  let speed_man := 6 -- km/hr
  let time_to_cross := 59.99520038396929 -- seconds
  let rel_speed := km_per_hr_to_m_per_s (relative_speed speed_train speed_man)
  length_of_train rel_speed time_to_cross = 400 :=
by
  sorry

end length_of_train_is_400_meters_l384_384286


namespace least_y_value_l384_384218

theorem least_y_value (y : ℝ) : 2 * y ^ 2 + 7 * y + 3 = 5 → y ≥ -2 :=
by
  intro h
  sorry

end least_y_value_l384_384218


namespace reciprocal_sum_ineq_l384_384771

noncomputable def r : ℕ → ℝ 
| 0       := 2
| (n + 1) := (List.prod (List.map r (List.range (n + 1)))) + 1

theorem reciprocal_sum_ineq (a : ℕ → ℝ) (n : ℕ) (h : (∑ i in Finset.range n, 1 / a i) < 1) :
  (∑ i in Finset.range n, 1 / a i) ≤ (∑ i in Finset.range n, 1 / r i) :=
by
  sorry

end reciprocal_sum_ineq_l384_384771


namespace part1_part2_part3_l384_384451

-- Definitions of the conditions
def condition1 : Prop :=
  ∃ P R : ℝ, ∃ (x y : ℝ), 
  ((x + 3) ^ 2 + y ^ 2 = 81) ∧ ((x - 3) ^ 2 + y ^ 2 = 1)

def condition2 : Prop :=
  ∀ Q x y : ℝ, Q ≠ 0 ∧ Q ∈ curve_C.1

def condition3 : Prop := 
  true -- O is always the origin

def condition4 : Prop :=
  ∀ (M N: ℝ), (line_OQ // line_F2).int_curve_C

-- Correct Answers Translated into Lean

-- Part Ⅰ: equation of curve C
theorem part1 (h1: condition1) : 
  (∀ P x y, (x^2 / 16 + y^2 / 7 = 1)):=
sorry

-- Part Ⅱ: Ratio of |MN| to |OQ|
theorem part2 (h1: condition1) (h2: condition2): 
  (∀ M N Q : ℝ, |MN| / |OQ|^2 = 1/2) :=
sorry

-- Part Ⅲ: Maximum value of S
theorem part3 (h1: condition1) (h2: condition2) (h4: condition4): 
  (∀ S : ℝ, max S = 2 * sqrt 7):=
sorry

end part1_part2_part3_l384_384451


namespace lockers_count_l384_384296

theorem lockers_count (cost_per_digit : ℕ) (total_cost : ℝ) :
  cost_per_digit = 3 →
  total_cost = 413.91 →
  ∃ n : ℕ, (let cost_1_digit := 9 * 1 * (cost_per_digit / 100 : ℝ),
                cost_2_digits := 90 * 2 * (cost_per_digit / 100 : ℝ),
                cost_3_digits := 900 * 3 * (cost_per_digit / 100 : ℝ),
                cost_4_digits := 2000 * 4 * (cost_per_digit / 100 : ℝ),
                cost_upto_2999 := cost_1_digit + cost_2_digits + cost_3_digits + cost_4_digits,
                remaining_cost := total_cost - cost_upto_2999,
                additional_digits := remaining_cost * 100 / cost_per_digit,
                additional_lockers := additional_digits / 4,
                total_lockers := 2999 + additional_lockers) in
                total_lockers = 3726) :=
by
  sorry

end lockers_count_l384_384296


namespace probability_prime_or_odd_l384_384913

-- Define the set of balls and their corresponding numbers
def balls : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}

-- Define the set of prime numbers among the balls
def primes : Finset ℕ := {2, 3, 5, 7}

-- Define the set of odd numbers among the balls
def odds : Finset ℕ := {1, 3, 5, 7}

-- Define the set of numbers that are either prime or odd
def primes_or_odds := primes ∪ odds

-- Calculate the probability as the ratio of the size of primes_or_odds set to the size of balls set
def probability := (primes_or_odds.card : ℚ) / balls.card

-- Statement that the probability is 5/7
theorem probability_prime_or_odd : probability = 5 / 7 := by
  sorry

end probability_prime_or_odd_l384_384913


namespace find_angle_l384_384876

noncomputable def angle_between_vectors (a b c : ℝ^3) 
  (norm_a : ‖a‖ = 2)
  (norm_b : ‖b‖ = 3)
  (norm_c : ‖c‖ = 4)
  (triple_product : a × (b × c) = 3 • b + 2 • c)
  (lin_independent : LinearIndependent ℝ ![a, b, c]) : ℝ :=
  Real.arccos (3 / 8)

theorem find_angle (a b c : ℝ^3)
  (norm_a : ‖a‖ = 2)
  (norm_b : ‖b‖ = 3)
  (norm_c : ‖c‖ = 4)
  (triple_product : a × (b × c) = 3 • b + 2 • c)
  (lin_independent : LinearIndependent ℝ ![a, b, c]) :
  angle_between_vectors a b c norm_a norm_b norm_c triple_product lin_independent = Real.arccos (3 / 8) :=
  sorry

end find_angle_l384_384876


namespace triangle_min_value_l384_384119

noncomputable def triangle_areas (AB BC CA PD PE PF : ℝ) : ℝ :=
  (8 / PD) + (9 / PE) + (7 / PF)

theorem triangle_min_value :
  let AB := 7 in
  let BC := 8 in
  let CA := 9 in
  ∃ a b c : ℕ, 
    let minimal_value := (triangle_areas AB BC CA (12*sqrt 5)) in
    let result := (24 * sqrt 5) / 5 in
    minimal_value = result ∧ a * b * c = 600 :=
by
  sorry

end triangle_min_value_l384_384119


namespace lucas_biking_speed_l384_384489

theorem lucas_biking_speed 
  (miguel_speed : ℝ)
  (sophie_factor : ℝ)
  (lucas_factor : ℝ)
  (H1 : miguel_speed = 6)
  (H2 : sophie_factor = 3 / 4)
  (H3 : lucas_factor = 4 / 3) : 
  let sophie_speed := sophie_factor * miguel_speed in
  let lucas_speed := lucas_factor * sophie_speed in
  lucas_speed = 6 := 
by
  sorry

end lucas_biking_speed_l384_384489


namespace probability_at_least_one_3_l384_384263

open Probability

noncomputable def fair_six_sided_die : Pmf ℤ := Pmf.uniform_of_finite 6 (λ (x : ℕ), x + 1)

def valid_tosses (X1 X2 X3 X4 : ℕ) := 
  (1, 1) ∈ fair_six_sided_die.to_finset ∧
  (1, 1) ∈ fair_six_sided_die.to_finset ∧
  (1, 1) ∈ fair_six_sided_die.to_finset ∧
  (1, 1) ∈ fair_six_sided_die.to_finset ∧
  (X1 + X2 + X3 = X4)

def at_least_one_three (X1 X2 X3 X4 : ℕ) := X1 = 3 ∨ X2 = 3 ∨ X3 = 3 ∨ X4 = 3

theorem probability_at_least_one_3 (h : valid_tosses X1 X2 X3 X4) : 
  probability fair_six_sided_die (at_least_one_three X1 X2 X3 X4) = 9 / 20 := 
sorry

end probability_at_least_one_3_l384_384263


namespace investment_plans_count_l384_384264

def num_projects : ℕ := 3
def num_cities : ℕ := 4

/-- The maximum number of projects that can be in one city. -/
def max_projects_per_city : ℕ := 2

/-- The number of different investment plans given the constraints provided. -/
def num_investment_plans : ℕ :=
  let scenario1 := (num_cities * (num_cities - 1) * (num_cities - 2)) / nat.factorial num_projects
  let scenario2 := num_cities * (num_cities - 1)
  scenario1 + scenario2

theorem investment_plans_count : num_investment_plans = 16 := by
  -- Proof of the theorem would go here.
  sorry

end investment_plans_count_l384_384264


namespace decreases_as_x_increases_l384_384618

theorem decreases_as_x_increases (f : ℝ → ℝ) (h₁ : f = λ x, x^2 + 1) (h₂ : f = λ x, -x^2 + 1) 
  (h₃ : f = λ x, 2x + 1) (h₄ : f = λ x, -2x + 1) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ > f x₂) ↔ f = λ x, -2x + 1 :=
sorry

end decreases_as_x_increases_l384_384618


namespace terminating_decimal_fraction_count_l384_384687

theorem terminating_decimal_fraction_count :
  (finset.filter (λ n : ℕ, 1 ≤ n ∧ n ≤ 180 ∧ 90 ∣ n) (finset.range 181)).card = 20 :=
by sorry

end terminating_decimal_fraction_count_l384_384687


namespace distance_focus_to_asymptote_of_hyperbola_l384_384374

theorem distance_focus_to_asymptote_of_hyperbola :
  ∀ (F : ℝ × ℝ), 
    (∃ c : ℝ, F = (c, 0) ∧ c^2 = 6) →
    (∃ a b : ℝ, a = 2 ∧ b = sqrt 2 ∧ (∀ x y : ℝ, y^2 = (b^2 / a^2) * x^2)) →
    ∃ d : ℝ, d = sqrt 2 :=
by
  sorry

end distance_focus_to_asymptote_of_hyperbola_l384_384374


namespace leave_zero_on_blackboard_l384_384193

theorem leave_zero_on_blackboard (n : ℕ) (h : n ≥ 2) : 
  ∃ seq : list ℕ, (∀ m ∈ seq, m ≠ 0) → (length seq = n → (n ≠ 3 → ∃ final_seq : list ℕ, final_seq = [0])) :=
sorry

end leave_zero_on_blackboard_l384_384193


namespace sign_selection_l384_384369

theorem sign_selection (i j k : Nat) (n : Nat) (a : Fin n → Nat)
  (h1 : ∀ t : Fin n, a t < i + 1)
  (h2 : ∃ I : Finset (Fin n), I.card = i ∧ (∀ t ∈ I, a t < j + 1))
  (h3 : ∃ J : Finset (Fin n), J.card = j ∧ (∀ t ∈ J, a t < k + 1)) :
  ∃ S : Fin n → Int, -k ≤ (Finset.univ.sum (λ t, S t * a t)) ∧ 
                       (Finset.univ.sum (λ t, S t * a t)) ≤ k := 
sorry

end sign_selection_l384_384369


namespace sufficient_not_necessary_condition_l384_384245

theorem sufficient_not_necessary_condition (x : ℝ) : (x > 0 → |x| = x) ∧ (|x| = x → x ≥ 0) :=
by
  sorry

end sufficient_not_necessary_condition_l384_384245


namespace terminating_decimal_count_l384_384747

theorem terminating_decimal_count : 
  (Finset.filter (λ n, n % 3 = 0) (Finset.range 181)).card = 60 := by
  sorry

end terminating_decimal_count_l384_384747


namespace seatingArrangements_l384_384617

-- Define the conditions as predicates on permutations of the 5 individuals
def noAdjacent (a b : Nat) (l : List Nat) : Prop :=
  ∀i, l.nth i = some a → (i > 0 → l.nth (i - 1) ≠ some b) ∧ (i < l.length - 1 → l.nth (i + 1) ≠ some b)

def validSeating (l : List Nat) : Prop :=
  noAdjacent 0 1 l ∧ noAdjacent 0 2 l ∧ noAdjacent 2 4 l ∧ noAdjacent 3 4 l

-- Define the assertion that there are exactly 4 valid ways
theorem seatingArrangements : (List.permutations [0, 1, 2, 3, 4]).countP validSeating = 4 :=
sorry

end seatingArrangements_l384_384617


namespace greatest_non_fiction_books_l384_384430

def is_prime (p : ℕ) := p > 1 ∧ (∀ d : ℕ, d ∣ p → d = 1 ∨ d = p)

theorem greatest_non_fiction_books (n f k : ℕ) :
  (n + f = 100 ∧ f = n + k ∧ is_prime k) → n ≤ 49 :=
by
  sorry

end greatest_non_fiction_books_l384_384430


namespace find_v4_l384_384204

-- Define the conditions given in the problem
variables (v1 v2 v3 v4 v5 v_avg : ℝ)
variables (h1 : v1 = 50)
variables (h2 : v2 = 62)
variables (h3 : v3 = 73)
variables (h5 : v5 = 40)
variables (h_avg : v_avg = 59)
variables (h_total : 5 * v_avg = 295)
variables (h_sum : v1 + v2 + v3 + v5 = 225)

-- Define the theorem
theorem find_v4 : v4 = 70 :=
by
  have h1 : v1 = 50 := by assumption
  have h2 : v2 = 62 := by assumption
  have h3 : v3 = 73 := by assumption
  have h5 : v5 = 40 := by assumption
  have h_avg : v_avg = 59 := by assumption
  have h_total : 5 * v_avg = 295 := by rw [h_avg, mul_comm]; exact rfl
  have h_sum : v1 + v2 + v3 + v5 = 225 := by rw [h1, h2, h3, h5]; exact rfl
  have h_v4 : v4 = (5 * v_avg) - (v1 + v2 + v3 + v5) := by tex
  rw [h_total, h_sum] at h_v4
  exact h_v4

end find_v4_l384_384204


namespace students_in_class_l384_384153

theorem students_in_class (n : ℕ) (T : ℕ) 
  (average_age_students : T = 16 * n)
  (staff_age : ℕ)
  (increased_average_age : (T + staff_age) / (n + 1) = 17)
  (staff_age_val : staff_age = 49) : n = 32 := 
by
  sorry

end students_in_class_l384_384153


namespace range_of_f_l384_384389

noncomputable def f (x : ℝ) : ℝ := Real.logb (1 / 2) (3 + 2 * x - x^2)

theorem range_of_f :
  {y : ℝ | ∃ x : ℝ, f x = y} = Set.Icc (-2) (Real.Infinity) :=
by
  sorry

end range_of_f_l384_384389


namespace max_value_proof_l384_384032

noncomputable def max_value_condition 
  (a : ℝ × ℝ) (b : ℝ × ℝ) (c : ℝ → ℝ × ℝ)
  (m n : ℝ) (α : ℝ) :=
  (m • a + n • b = 2 • c α) → ∃ (M : ℝ), M = 36 ∧
  ∀ (alpha : ℝ), (let (x, y) := m • a + n • b in (x - 4)^2 + y^2 ≤ M)

def given_vectors := (1 : ℝ, -1 : ℝ)

def given_vectors2 := (1 : ℝ, 1 : ℝ)

def given_vector_function (α : ℝ) := (real.sqrt 2 * real.cos α, real.sqrt 2 * real.sin α)

-- Original function to rewrite in Lean 4
theorem max_value_proof : 
  max_value_condition (1, -1) (1, 1) 
  (λ α => (real.sqrt 2 * real.cos α, real.sqrt 2 * real.sin α)) := 
begin
  intros m n α h,
  -- Maximal value of the expression
  use 36,
  split,
  { refl },
  { intros α,
    -- Proving the maximal value constraint here
    sorry
  }
end

end max_value_proof_l384_384032


namespace terminating_decimals_count_l384_384722

theorem terminating_decimals_count :
  ∃ (count : ℕ), count = 60 ∧
  ∀ (n : ℕ), 1 ≤ n ∧ n ≤ 180 →
  (∃ m : ℕ, m * 180 = n * (2^2 * 5)) → 60 = count := sorry

end terminating_decimals_count_l384_384722


namespace rat_op_neg2_3_rat_op_4_neg2_eq_neg2_4_l384_384349

namespace RationalOperation

-- Definition of the operation ⊗ for rational numbers
def rat_op (a b : ℚ) : ℚ := a * b - a - b - 2

-- Proof problem 1: (-2) ⊗ 3 = -9
theorem rat_op_neg2_3 : rat_op (-2) 3 = -9 :=
by
  sorry

-- Proof problem 2: 4 ⊗ (-2) = (-2) ⊗ 4
theorem rat_op_4_neg2_eq_neg2_4 : rat_op 4 (-2) = rat_op (-2) 4 :=
by
  sorry

end RationalOperation

end rat_op_neg2_3_rat_op_4_neg2_eq_neg2_4_l384_384349


namespace fraction_claim_dislike_but_enjoy_l384_384629

-- Definitions based on conditions
def total_students := 100
def enjoy_reading := 0.7 * total_students
def dislike_reading := 0.3 * total_students

def pretend_dislike := 0.3 * enjoy_reading
def honest_dislike := 0.75 * dislike_reading

-- Main statement to prove
theorem fraction_claim_dislike_but_enjoy : 
  pretend_dislike / (pretend_dislike + honest_dislike) = 21 / 43 := by
  -- Proof omitted
  sorry

end fraction_claim_dislike_but_enjoy_l384_384629


namespace determinant_of_matrix_is_zero_l384_384303

theorem determinant_of_matrix_is_zero :
  let M := matrix
    ![ ![real.sin 1, real.cos 1, real.cos 2],
       ![real.sin 4, real.cos 4, real.cos 5],
       ![real.sin 7, real.cos 7, real.cos 8] ]
  det M = 0 :=
by sorry

end determinant_of_matrix_is_zero_l384_384303


namespace CannotDetermineDraculaStatus_l384_384861

variable (Transylvanian_is_human : Prop)
variable (Dracula_is_alive : Prop)
variable (Statement : Transylvanian_is_human → Dracula_is_alive)

theorem CannotDetermineDraculaStatus : ¬ (∃ (H : Prop), H = Dracula_is_alive) :=
by
  sorry

end CannotDetermineDraculaStatus_l384_384861


namespace intersection_A_B_l384_384372

def A : Set ℝ := { x | Real.sqrt x ≤ 3 }
def B : Set ℝ := { x | x^2 ≤ 9 }

theorem intersection_A_B : A ∩ B = { x | 0 ≤ x ∧ x ≤ 3 } :=
by
  sorry

end intersection_A_B_l384_384372


namespace intersection_A_B_is_C_l384_384780

def set_A : Set ℝ := { x | -2 < x ∧ x < 4 }
def set_B : Set ℝ := { x | log 2 x < 1 }
def set_C : Set ℝ := { x | 0 < x ∧ x < 2 }

theorem intersection_A_B_is_C : (set_A ∩ set_B) = set_C := sorry

end intersection_A_B_is_C_l384_384780


namespace findMinorAxisLength_l384_384613

def isEllipseThroughFivePoints (p1 p2 p3 p4 p5 : (ℝ × ℝ)) : Prop :=
  ∃ a b c d e f : ℝ, (a ≠ 0 ∨ b ≠ 0) ∧
  ∀ (x y : ℝ), (x, y) ∈ {p1, p2, p3, p4, p5} →
  a * x^2 + b * x * y + c * y^2 + d * x + e * y + f = 0

def areAxesParallelToCoordinateAxes (a b c : ℝ) : Prop :=
  b = 0 ∧ a ≠ c

noncomputable def ellipseMinorAxis (a b c d e f : ℝ) : ℝ :=
  if h : areAxesParallelToCoordinateAxes a b c then
    2 * sqrt (1 / (abs c))
  else 0

theorem findMinorAxisLength :
  isEllipseThroughFivePoints (-2, -1) (0, 0) (0, 3) (4, 0) (4, 3) ∧
  ∃ a b c d e f : ℝ, areAxesParallelToCoordinateAxes a b c ∧ 
  ∀ (x y : ℝ), (x, y) ∈ {(-2, -1), (0, 0), (0, 3), (4, 0), (4, 3)} →
    a * x^2 + b * x * y + c * y^2 + d * x + e * y + f = 0
  → ellipseMinorAxis a b c d e f = 2 * sqrt 3 :=
sorry

end findMinorAxisLength_l384_384613


namespace termite_not_collapsing_l384_384232

theorem termite_not_collapsing (h_termites : ℚ) 
  (h_collapsing_of_termites : ℚ) : 
  h_termites = 1/3 ∧ h_collapsing_of_termites = 4/7 → (1/3 - (1/3 * 4/7)) = 3/21 :=
by
  intro h
  cases h with h_termites_eq h_collapsing_of_termites_eq
  have h_collapsing_eq : (1/3 * 4/7) = 4/21,
    sorry,
  rw h_termites_eq,
  rw h_collapsing_eq,
  simp,
  sorry

end termite_not_collapsing_l384_384232


namespace probability_of_irrational_l384_384289

def is_rational (x : Real) := ∃ a b : Int, b ≠ 0 ∧ x = a / b
def is_irrational (x : Real) := ¬ is_rational x

theorem probability_of_irrational :
  let S := {x : ℝ | x = 22 / 7 ∨ x = -2023 ∨ x = Real.pi / 3 ∨ x = Real.sqrt 2}
  let irr := {x ∈ S | is_irrational x}
  |irr| / |S| = 1 / 2 := by
  sorry

end probability_of_irrational_l384_384289


namespace Jonah_paid_commensurate_l384_384867

def price_per_pineapple (P : ℝ) :=
  let number_of_pineapples := 6
  let rings_per_pineapple := 12
  let total_rings := number_of_pineapples * rings_per_pineapple
  let price_per_4_rings := 5
  let price_per_ring := price_per_4_rings / 4
  let total_revenue := total_rings * price_per_ring
  let profit := 72
  total_revenue - number_of_pineapples * P = profit

theorem Jonah_paid_commensurate {P : ℝ} (h : price_per_pineapple P) :
  P = 3 :=
  sorry

end Jonah_paid_commensurate_l384_384867


namespace employee_overtime_hours_l384_384624

theorem employee_overtime_hours (h1 : ∀ x, 0 ≤ x → Integer.floor (x + 0.5) = x) :
    ∃ (overtime_hours : ℕ), 
    let regular_hours := 40
    let regular_rate := 11.25
    let overtime_rate := 16
    let gross_pay := 622
    let regular_pay := (regular_hours : ℕ) * regular_rate
    let overtime_pay := gross_pay - regular_pay
    overtime_hours = Integer.floor (overtime_pay / overtime_rate) + 0.5 ∧ 
    overtime_hours = 11 := 
sorry

end employee_overtime_hours_l384_384624


namespace function_increasing_interval_l384_384952

theorem function_increasing_interval :
  ∀ x : ℝ, x ∈ Icc (0 : ℝ) Real.pi →
  ∃ (a b : ℝ), a = Real.pi / 3 ∧ b = 5 * Real.pi / 6 ∧ 
  x ∈ Icc a b ∧ deriv (λ x, 2 * Real.sin (Real.pi / 6 - 2 * x)) x > 0 :=
begin
  intro x,
  intro hx,
  use (Real.pi / 3),
  use (5 * Real.pi / 6),
  split, exact rfl,
  split, exact rfl,
  split,
  { rw Icc at hx,
    split;
    linarith,
  },
  sorry  -- Proof to be filled in later
end

end function_increasing_interval_l384_384952


namespace eq_of_triangle_l384_384859

variables {A B C K E : Type} [euclidean_geometry A B C]
variables (ABC : triangle A B C) (K_midpoint : is_midpoint K B C) (AE_altitude : is_altitude A E B C)

theorem eq_of_triangle (h₁ : ∠ABC = 2 * ∠ACB) : AB = 2 * EK :=
by
  sorry

end eq_of_triangle_l384_384859


namespace problem_solved_by_three_girls_and_three_boys_l384_384581

theorem problem_solved_by_three_girls_and_three_boys
  (G B : ℕ)
  (n : ℕ)
  (conditions : ∀ i : Fin n, (∃ k : Fin G, k ≠ i → (solved(k, problem(i)) → solved(j, problem(i))))
  (h1 : G = 2221)
  (h2 : B = 21)
  (h3 : ∀ i : Fin (G + B), ∀ j : Fin (G + B), i ≠ j → solved(j, problem(i)) → solved(i, problem(i))
  (h4 : ∀ p: Fin n, (#solved(p) ≤ 6))

  (h5 : ∀ i : Fin 2221, ∃ j : Fin 21, solved(i, j) ∧ solved(j, i)) :
∃ p, (#solved_girls(p) ≥ 3) (solved_boys(p) ≥ 3) :=
by 
  sorry

end problem_solved_by_three_girls_and_three_boys_l384_384581


namespace domain_of_sqrt_function_l384_384942

theorem domain_of_sqrt_function (f : ℝ → ℝ) (x : ℝ) 
  (h1 : ∀ x, (1 / (Real.log x) - 2) ≥ 0) 
  (h2 : ∀ x, Real.log x ≠ 0) : 
  (1 < x ∧ x ≤ Real.sqrt 10) ↔ (∀ x, 0 < Real.log x ∧ Real.log x ≤ 1 / 2) := 
  sorry

end domain_of_sqrt_function_l384_384942


namespace angle_aie_in_triangle_l384_384079

theorem angle_aie_in_triangle (A B C D E F I : Type) [inhabited A]
  (h1 : ∠ACB = 50) (h2 : BC = BD) (h3 : incenter I ABC AD BE CF)
  (h4 : ∠BCD = ∠BDC) : ∠AIE = 65 :=
by
  sorry

end angle_aie_in_triangle_l384_384079


namespace average_of_five_integers_l384_384512

theorem average_of_five_integers :
  ∃ (k m r s t : ℕ), k < m ∧ m < r ∧ r < s ∧ s < t ∧ t = 40 ∧ r = 23 ∧ (k + m + r + s + t) / 5 = 18 :=
by {
  use [1, 2, 23, 24, 40],
  split,
  repeat { norm_num <|> linarith }
}

end average_of_five_integers_l384_384512


namespace factor_as_complete_square_l384_384006

theorem factor_as_complete_square (k : ℝ) : (∃ a : ℝ, x^2 + k*x + 9 = (x + a)^2) ↔ k = 6 ∨ k = -6 := 
sorry

end factor_as_complete_square_l384_384006


namespace ξ_is_martingale_l384_384578

variables {Ω : Type*} {n : ℕ} 
variables (P : Ω → Prop)
variables (𝒟 : ℕ → set (set Ω))
variables (η : ℕ → Ω → ℝ)
variables [probability_space Ω]

-- Conditions
-- Assume the sequence of partitions and their properties
axiom D_seq : ∀ k, measurable_space.sets (𝒟 k) ⊆ measurable_space.sets (𝒟 (k + 1))
axiom D0 : 𝒟 0 = {Ω}

-- Assume η_k is 𝒟_k-measurable
axiom η_measurable : ∀ k (h : 1 ≤ k ≤ n), @measurable Ω ℝ (𝒟 k) (borel ℝ) (η k)

-- Define ξ sequence
noncomputable def ξ (k : ℕ) : Ω → ℝ :=
  ∑ l in finset.range (k + 1), η l - (conditional_expectation (𝒟 (l - 1)) (η l))

-- Theorem : Show that ξ is a martingale
theorem ξ_is_martingale : (martingale ξ 𝒟) :=
sorry

end ξ_is_martingale_l384_384578


namespace field_ratio_l384_384520

theorem field_ratio (length_multiple_width : ∃ k : ℕ, 16 = k * w)
                    (pond_side : pond_side = 4)
                    (pond_area_ratio : pond_area / field_area = 1 / 8)
                    (field_length : l = 16) :
                    length_ratio : l / w = 2 :=
by
  -- l and w are the length and width of the field respectively
  let field_area := 16 * w
  let pond_area := 4 * 4
  have pond_area_eq : pond_area = 16 := by sorry
  have field_area_eq : field_area = 128 := by sorry
  have width_eq : w = 128 / 16 := by sorry
  have ratio_eq : l / w = 16 / 8 := by sorry
  exact (field_ratio 2)

end field_ratio_l384_384520


namespace problem_statement_l384_384797

def f (x : ℝ) : ℝ :=
  if x < 0 then 1 - x^2
  else x^2 - x - 1

theorem problem_statement : f (-1) + f (2) = 1 :=
by sorry

end problem_statement_l384_384797


namespace integral_sin_8_eq_32pi_l384_384300

noncomputable def integral_sin_8 : Real :=
  ∫ x in -Real.pi/2..0, 2^8 * Real.sin(x)^8

theorem integral_sin_8_eq_32pi :
  integral_sin_8 = 32 * Real.pi :=
  by sorry

end integral_sin_8_eq_32pi_l384_384300


namespace part_I_part_II_l384_384812

variables {α : Type*} [LinearOrder α]

def setA (a : α) : Set α := { x : α | a - 1 < x ∧ x < 2 * a + 1 }
def setB : Set α := { x : α | 0 < x ∧ x < 1 }

theorem part_I (a : α) (h : a = (1 : α) / 2) : 
  setA a ∩ setB = { x : α | 0 < x ∧ x < 1 } :=
by
  sorry

theorem part_II : (∀ a : α, setA a ≠ ∅) → 
  (setA a ∩ setB = ∅) → 
  -2 < a ∧ a ≤ (1 : α) / 2 ∨ a ≥ 2 :=
by
  sorry

end part_I_part_II_l384_384812


namespace solve_inequality_l384_384928

theorem solve_inequality (x : ℝ) :
  (x - 2) / (x + 5) ≤ 1 / 2 ↔ x ∈ Set.Ioc (-5 : ℝ) 9 :=
by
  sorry

end solve_inequality_l384_384928


namespace inverse_matrix_proof_l384_384334

variable (A : Matrix (Fin 2) (Fin 2) ℤ)
variable (B : Matrix (Fin 2) (Fin 2) ℤ)
variable (zeroMatrix : Matrix (Fin 2) (Fin 2) ℤ := ![(0, 0), (0, 0)])

-- Condition: The given matrices
def matrixA := ![(5, -3), (-2, 1)]
def matrixB := ![(-1, -3), (-2, -5)]

-- Property to prove: matrixB is the inverse of matrixA
theorem inverse_matrix_proof : 
  (∀ A : Matrix (Fin 2) (Fin 2) ℤ, A = matrixA) →
  (∀ B : Matrix (Fin 2) (Fin 2) ℤ, B = matrixB) →
  (B ⬝ A = 1) := 
  by sorry

end inverse_matrix_proof_l384_384334


namespace parallelepiped_in_sphere_conditions_l384_384261

noncomputable def parallelepiped_edges : ℝ := sorry
noncomputable def parallelepiped_height : ℝ := sorry

theorem parallelepiped_in_sphere_conditions :
  let V_s := (4 / 3) * π * (1 : ℝ)^3 in
  let A_s := 4 * π * (1 : ℝ)^2 in
  let a := parallelepiped_edges in
  let h := parallelepiped_height in
  a ≈ 1 ∧ h ≈ (2 * π / 3) ∧ 
  (a^2 * h = (2 / 3) * π) ∧ 
  (2 * a^2 + 4 * a * h = 2 * π) := 
by
  sorry

end parallelepiped_in_sphere_conditions_l384_384261


namespace acute_triangle_exists_l384_384343

theorem acute_triangle_exists {a1 a2 a3 a4 a5 : ℝ} 
  (h1 : a1 + a2 > a3) (h2 : a1 + a3 > a2) (h3 : a2 + a3 > a1)
  (h4 : a2 + a3 > a4) (h5 : a3 + a4 > a2) (h6 : a2 + a4 > a3)
  (h7 : a3 + a4 > a5) (h8 : a4 + a5 > a3) (h9 : a3 + a5 > a4) : 
  ∃ (t1 t2 t3 : ℝ), (t1 + t2 > t3) ∧ (t1 + t3 > t2) ∧ (t2 + t3 > t1) ∧ (t3 ^ 2 < t1 ^ 2 + t2 ^ 2) :=
sorry

end acute_triangle_exists_l384_384343


namespace evaluated_expression_l384_384649

noncomputable def evaluation_problem (x a y z c d : ℝ) : ℝ :=
  (2 * x^3 - 3 * a^4) / (y^2 + 4 * z^5) + c^4 - d^2

theorem evaluated_expression :
  evaluation_problem 0.66 0.1 0.66 0.1 0.066 0.1 = 1.309091916 :=
by
  sorry

end evaluated_expression_l384_384649


namespace sum_of_repeating_decimals_l384_384314

noncomputable def repeating_decimal_sum_to_fraction : Prop :=
  let x := 0.2 + 0.05 / 100 + 0.0003 / 10000 in
  let frac_x := 2 / 9 + 5 / 99 + 3 / 9999 in
  x = frac_x

theorem sum_of_repeating_decimals : repeating_decimal_sum_to_fraction := sorry

end sum_of_repeating_decimals_l384_384314


namespace probability_of_rain_on_at_most_3_days_is_0_707_l384_384958

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

noncomputable def probability_rain_on_k_days (p q : ℚ) (n k : ℕ) : ℚ :=
  (binomial_coefficient n k) * (p^k) * (q^(n-k))

noncomputable def total_probability_of_rain (p : ℚ) (n : ℕ) : ℚ :=
  let q := 1 - p
  (probability_rain_on_k_days p q n 0) + (probability_rain_on_k_days p q n 1) +
  (probability_rain_on_k_days p q n 2) + (probability_rain_on_k_days p q n 3)

theorem probability_of_rain_on_at_most_3_days_is_0_707 : 
  (total_probability_of_rain (2/10 : ℚ) 31) ≈ (707/1000 : ℚ) := 
by 
  -- skipping the proof
  sorry

end probability_of_rain_on_at_most_3_days_is_0_707_l384_384958


namespace infinite_perfect_square_set_l384_384459

def d (n : ℕ) : ℕ := ∑ i in (finset.range (n+1)).filter (λ i, n % i = 0), i

def φ (n : ℕ) : ℕ := (finset.range (n+1)).filter (λ i, nat.gcd n i = 1).card

theorem infinite_perfect_square_set :
  set.infinite {n : ℕ | ∃ k : ℕ, d(n) * φ(n) = k^2} :=
sorry

end infinite_perfect_square_set_l384_384459


namespace max_sides_convex_with_three_obtuse_l384_384260

open Real

-- Definitions for obtuse angles and internal angles sum in convex polygons
def is_obtuse (θ : ℝ) : Prop :=
  90 < θ ∧ θ < 180

def is_convex_polygon (n : ℕ) (angles : Fin n → ℝ) : Prop :=
  ∀ i, angles i > 0 ∧ angles i < 180 ∧ (Finset.univ.sum angles) = (n - 2) * 180

-- Main theorem stating the condition that convex polygon has exactly three obtuse angles
theorem max_sides_convex_with_three_obtuse (n : ℕ) (angles : Fin n → ℝ) (h_convex : is_convex_polygon n angles)
  (h_obtuse : ∃ i j k : Fin n, i ≠ j ∧ i ≠ k ∧ j ≠ k ∧ is_obtuse (angles i) ∧ is_obtuse (angles j) ∧ is_obtuse (angles k) ∧
              ∀ l : Fin n, (l ≠ i ∧ l ≠ j ∧ l ≠ k) → ¬ is_obtuse (angles l)) : n ≤ 6 :=
by sorry

end max_sides_convex_with_three_obtuse_l384_384260


namespace product_divides_l384_384779

open Nat

-- Define the sequence {x_n}
def seq (a b : ℕ) : ℕ → ℕ
| 0       := 0
| 1       := 1
| (2*n)   := a * seq (2*n - 1) - seq (2*n - 2)
| (2*n+1) := b * seq (2*n) - seq (2*n - 1)

-- The main theorem statement
theorem product_divides
  (a b : ℕ) (h_b : b > 1) (m n : ℕ) (h_m : m > 0) (h_n : n > 0) :
  (\prod i in range (1, m + 1), seq a b i) ∣ (\prod i in range (n + 1, n + m + 1), seq a b i) :=
by {
  sorry -- Proof to be done
}

end product_divides_l384_384779


namespace dan_must_exceed_48_mph_l384_384235

theorem dan_must_exceed_48_mph :
  ∀ (d v_c t_d v_d : ℝ),
  (d = 120) → (v_c = 30) → (t_d = 1.5) → (v_d > 48) :=
by
  intros d v_c t_d v_d
  assume h_d h_vc h_td
  sorry

end dan_must_exceed_48_mph_l384_384235


namespace num_ways_students_accepted_l384_384251

theorem num_ways_students_accepted :
  ∃ (ways : ℕ), ways = 36 ∧ 
  (∀ (A B C : set string), 
  A ⊆ {"student1", "student2", "student3", "student4"} ∧ 
  B ⊆ {"student1", "student2", "student3", "student4"} ∧ 
  C ⊆ {"student1", "student2", "student3", "student4"} ∧ 
  A ∪ B ∪ C = {"student1", "student2", "student3", "student4"} ∧ 
  A ≠ ∅ ∧ B ≠ ∅ ∧ C ≠ ∅ ∧ 
  ∀ (x : string), x ∈ A → x ∉ B ∧ x ∉ C ∧ (x ∈ B → x ∉ A ∧ x ∉ C) ∧ (x ∈ C → x ∉ A ∧ x ∉ B)) :=
begin
  -- proof goes here
  sorry
end

end num_ways_students_accepted_l384_384251


namespace function_range_is_one_to_two_l384_384661

noncomputable def range_of_function := {y : ℝ | ∃ x : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ y = sqrt x + sqrt (3 - 3 * x)}

theorem function_range_is_one_to_two : range_of_function = {y : ℝ | 1 ≤ y ∧ y ≤ 2} := 
sorry

end function_range_is_one_to_two_l384_384661


namespace find_a6_l384_384444

variable {a : ℕ → ℤ} -- Assume we have a sequence of integers
variable (d : ℤ) -- Common difference of the arithmetic sequence

-- Conditions
axiom h1 : a 3 = 7
axiom h2 : a 5 = a 2 + 6

-- Define arithmetic sequence property
def arithmetic_seq (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ (n : ℕ), a (n + 1) = a n + d

-- Theorem to prove
theorem find_a6 (h1 : a 3 = 7) (h2 : a 5 = a 2 + 6) (h3 : arithmetic_seq a d) : a 6 = 13 :=
by
  sorry

end find_a6_l384_384444


namespace true_propositions_l384_384622

theorem true_propositions :
  (∀ (x y : ℝ), x * y = 1 → (x = 1 / y ∧ y = 1 / x)) ∧                                   -- Proposition 1
  (¬ ∀ (q : Type) [field q], ∀ (a b c d : q), (a = b ∧ b = c ∧ c = d ∧ d = a) → (a = b ∧ a * b = a * d)) ∧  -- Proposition 2
  (¬ ∀ (parallelogram trapezoid : Type), (parallelogram → trapezoid)) ∧                      -- Proposition 3
  (∀ (a b : ℝ) (c : ℝ), c ≠ 0 → a * c^2 > b * c^2 → a > b)                                   -- Proposition 4 
:=
sorry

end true_propositions_l384_384622


namespace number_of_divisors_l384_384038

-- Defining the given number and its prime factorization as a condition.
def given_number : ℕ := 90

-- Defining the prime factorization.
def prime_factorization (n : ℕ) : List (ℕ × ℕ) :=
  if n = 90 then [(2, 1), (3, 2), (5, 1)] else []

-- The statement to prove that the number of positive divisors of 90 is 12.
theorem number_of_divisors (n : ℕ) (pf : List (ℕ × ℕ)) :
  n = 90 → pf = [(2, 1), (3, 2), (5, 1)] →
  (pf.map (λ p, p.2 + 1)).prod = 12 :=
by
  intros hn hpf
  rw [hn, hpf]
  simp
  sorry

end number_of_divisors_l384_384038


namespace find_nabla_l384_384407

theorem find_nabla : ∀ (nabla : ℤ), 5 * (-4) = nabla + 2 → nabla = -22 :=
by
  intros nabla h
  sorry

end find_nabla_l384_384407


namespace terminating_decimal_integers_count_l384_384735

/-- For how many integer values of n between 1 and 180 inclusive does the decimal 
representation of n / 180 terminate? We need to calculate the number of integers n such 
that n includes at least 9 as a factor. The answer is 20. -/
theorem terminating_decimal_integers_count :
  { n : ℤ | 1 ≤ n ∧ n ≤ 180 ∧ 9 ∣ n }.to_finset.card = 20 := 
by sorry

end terminating_decimal_integers_count_l384_384735


namespace terminating_decimals_count_l384_384721

theorem terminating_decimals_count :
  ∃ (count : ℕ), count = 60 ∧
  ∀ (n : ℕ), 1 ≤ n ∧ n ≤ 180 →
  (∃ m : ℕ, m * 180 = n * (2^2 * 5)) → 60 = count := sorry

end terminating_decimals_count_l384_384721


namespace same_incenter_l384_384244

variables {P O A B Q C D I : Type}

-- Conditions
def is_point_on_exterior (P : Type) (O : Type) : Prop := sorry
def is_tangent (P A : Type) (circle_center O : Type) : Prop := sorry
def is_intersection_point (Q : Type) (line_PO line_AB : Type) : Prop := sorry
def is_chord_through (CD : Type) (Q : Type) : Prop := sorry
def is_incenter (I : Type) (triangle : Type) : Prop := sorry

theorem same_incenter 
  {P O A B Q C D I : Type}
  (h1 : is_point_on_exterior P O)
  (h2 : is_tangent P A O ∧ is_tangent P B O)
  (h3 : is_intersection_point Q P O ∧ is_intersection_point Q A B)
  (h4 : is_chord_through C D Q)
  (h5 : is_incenter I (P, A, B))
  (h6 : is_incenter I (P, C, D))
  : is_incenter I (P, A, B) ∧ is_incenter I (P, C, D) := 
sorry

end same_incenter_l384_384244


namespace convince_jury_l384_384546

def not_guilty : Prop := sorry  -- definition indicating the defendant is not guilty
def not_liar : Prop := sorry    -- definition indicating the defendant is not a liar
def innocent_knight_statement : Prop := sorry  -- statement "I am an innocent knight"

theorem convince_jury (not_guilty : not_guilty) (not_liar : not_liar) : innocent_knight_statement :=
sorry

end convince_jury_l384_384546


namespace inverse_of_matrix_l384_384330

noncomputable def my_matrix := matrix ([[5, -3], [-2, 1]])

theorem inverse_of_matrix :
  ∃ (M_inv : matrix ℕ ℕ ℝ), (my_matrix.det ≠ 0) ∧ (my_matrix * M_inv = 1) → M_inv = matrix ([[ -1, -3 ], [-2, -5 ]]) :=
by
  sorry

end inverse_of_matrix_l384_384330


namespace decreasing_y_as_x_increases_l384_384620

theorem decreasing_y_as_x_increases :
  (∀ x1 x2, x1 < x2 → (-2 * x1 + 1) > (-2 * x2 + 1)) ∧
  ¬ (∀ x1 x2, x1 < x2 → (x1^2 + 1) > (x2^2 + 1)) ∧
  ¬ (∀ x1 x2, x1 < x2 → (-x1^2 + 1) > (-x2^2 + 1)) ∧
  ¬ (∀ x1 x2, x1 < x2 → (2 * x1 + 1) > (2 * x2 + 1)) :=
by
  sorry

end decreasing_y_as_x_increases_l384_384620


namespace decreasing_function_range_l384_384795

theorem decreasing_function_range (a : ℝ) 
  (hf : ∀x y : ℝ, x < y → f x ≥ f y)
  (f : ℝ → ℝ := λ x, if x < 0 then x^2 - (2*a-1)*x + 1 else (a-3)*x + a) :
  1/2 ≤ a ∧ a ≤ 1 :=
sorry

end decreasing_function_range_l384_384795


namespace L_div_l_ge_two_l384_384980

noncomputable def inscribed_square_side_length (T : Triangle) : Real :=
  -- Placeholder for the actual maximum side length of a square inscribed in triangle T
  sorry

noncomputable def circumscribed_square_side_length (T : Triangle) : Real :=
  -- Placeholder for the actual minimum side length of a square circumscribed around triangle T
  sorry

theorem L_div_l_ge_two (T : Triangle) : 
  let l := inscribed_square_side_length T,
  let L := circumscribed_square_side_length T in
  L / l ≥ 2 :=
by
  sorry

end L_div_l_ge_two_l384_384980


namespace terminating_decimal_integers_count_l384_384736

/-- For how many integer values of n between 1 and 180 inclusive does the decimal 
representation of n / 180 terminate? We need to calculate the number of integers n such 
that n includes at least 9 as a factor. The answer is 20. -/
theorem terminating_decimal_integers_count :
  { n : ℤ | 1 ≤ n ∧ n ≤ 180 ∧ 9 ∣ n }.to_finset.card = 20 := 
by sorry

end terminating_decimal_integers_count_l384_384736


namespace average_weight_increase_l384_384938

theorem average_weight_increase 
  (A : ℝ) 
  (P1 : 75) 
  (P2 : 99.5) 
  (n : ℝ := 7) :
  let old_average := A,
      new_total_weight := 7 * A - P1 + P2,
      new_average := new_total_weight / n
  in new_average - old_average = 3.5 :=
by
  sorry

end average_weight_increase_l384_384938


namespace corner_sum_possible_values_l384_384241

-- Define the problem parameters
def Board (m n : ℕ) := (fin (m + 1) × fin (n + 1)) → bool

def is_checkerboard_pattern (b : Board 2016 2017) : Prop :=
  ∀ (i : fin 2016) (j : fin 2017),
    (i.1 + j.1) % 2 = 0 → b (i, j) = tt ∧ b (⟨i + 1, _⟩, j) = tt ∧ b (i, ⟨j + 1, _⟩) = ff ∧ b (⟨i + 1, _⟩, ⟨j + 1, _⟩) = ff ∧ 
    (i.1 + j.1) % 2 = 1 → b (i, j) = ff ∧ b (⟨i + 1, _⟩, j) = ff ∧ b (i, ⟨j + 1, _⟩) = tt ∧ b (⟨i + 1, _⟩, ⟨j + 1, _⟩) = tt

def sum_is_even (s : set (fin 2017 × fin 2018)) : Prop :=
  ∃ (n evens odds : ℕ),
  set.finite s ∧
  (∀ x ∈ s, x.fst % 2 = 0 → evens += 1) ∧
  (∀ x ∈ s, x.fst % 2 = 1 → odds += 1) ∧
  (evens + odds).even

def sum_is_odd (s : set (fin 2017 × fin 2018)) : Prop :=
  ∃ (n evens odds : ℕ),
  set.finite s ∧
  (∀ x ∈ s, x.fst % 2 = 0 → evens += 1) ∧
  (forall x ∈ s, x.fst % 2 = 1 → odds += 1) ∧
  (evens + odds).odd

def corner_sum (b : Board 2016 2017) : ℕ :=
  (if b (⟨0, sorry⟩, ⟨0, sorry⟩) then 1 else 0) +
  (if b (⟨0, sorry⟩, ⟨2017, sorry⟩) then 1 else 0) +
  (if b (⟨2016, sorry⟩, ⟨0, sorry⟩) then 1 else 0) +
  (if b (⟨2016, sorry⟩, ⟨2017, sorry⟩) then 1 else 0)

theorem corner_sum_possible_values (b : Board 2016 2017) (h : is_checkerboard_pattern b)  :
  corner_sum b = 0 ∨ corner_sum b = 2 ∨ corner_sum b = 4 :=
sorry

end corner_sum_possible_values_l384_384241


namespace smoothies_count_l384_384495

theorem smoothies_count :
  let initial_strawberries := 28 
  let picked_strawberries := 35
  let strawberries_per_smoothie := 7.5 
  let total_strawberries := initial_strawberries + picked_strawberries
  let number_of_whole_smoothies := total_strawberries / strawberries_per_smoothie
  ∃ number_of_smoothies : ℕ, number_of_smoothies = 8 :=
begin
  let initial_strs := initial_strawberries,
  let picked_strs := picked_strawberries,
  let strs_per_smth := strawberries_per_smoothie,
  let total_strs := initial_strs + picked_strs,
  let n_smth := (total_strs / strs_per_smth).toNat,
  use n_smth,
  sorry
end

end smoothies_count_l384_384495


namespace terminating_decimals_count_l384_384732

theorem terminating_decimals_count : 
  let N := 180 in
  let count_terminating_decimals := (List.range' 1 N).filter (λ n, n % 3 = 0) in
  count_terminating_decimals.length = 60 :=
by
  let N := 180
  let count_terminating_decimals := (List.range' 1 N).filter (λ n, n % 3 = 0)
  sorry

end terminating_decimals_count_l384_384732


namespace vector_sum_correct_l384_384339

def vec1 : Fin 3 → ℤ := ![-7, 3, 5]
def vec2 : Fin 3 → ℤ := ![4, -1, -6]
def vec3 : Fin 3 → ℤ := ![1, 8, 2]
def expectedSum : Fin 3 → ℤ := ![-2, 10, 1]

theorem vector_sum_correct :
  (fun i => vec1 i + vec2 i + vec3 i) = expectedSum := 
by
  sorry

end vector_sum_correct_l384_384339


namespace part1_double_root_equation_part2_value_m_squared_2m_2_part3_value_m_l384_384827

-- Part 1: Is x^2 - 3x + 2 = 0 a "double root equation"?
theorem part1_double_root_equation :
    ∃ (x₁ x₂ : ℝ), (x₁ ≠ x₂ ∧ x₁ * 2 = x₂) 
              ∧ (x^2 - 3 * x + 2 = 0) :=
sorry

-- Part 2: Given (x - 2)(x - m) = 0 is a "double root equation", find value of m^2 + 2m + 2.
theorem part2_value_m_squared_2m_2 (m : ℝ) :
    ∃ (v : ℝ), v = m^2 + 2 * m + 2 ∧ 
          (m = 1 ∨ m = 4) ∧
          (v = 5 ∨ v = 26) :=
sorry

-- Part 3: Determine m such that x^2 - (m-1)x + 32 = 0 is a "double root equation".
theorem part3_value_m (m : ℝ) :
    x^2 - (m - 1) * x + 32 = 0 ∧ 
    (m = 13 ∨ m = -11) :=
sorry

end part1_double_root_equation_part2_value_m_squared_2m_2_part3_value_m_l384_384827


namespace total_cases_after_three_weeks_l384_384896

-- Definitions and conditions directly from the problem
def week1_cases : ℕ := 5000
def week2_cases : ℕ := week1_cases / 2
def week3_cases : ℕ := week2_cases + 2000
def total_cases : ℕ := week1_cases + week2_cases + week3_cases

-- The theorem to prove
theorem total_cases_after_three_weeks :
  total_cases = 12000 := 
by
  -- Sorry allows us to skip the actual proof
  sorry

end total_cases_after_three_weeks_l384_384896


namespace task_assignment_count_l384_384538

theorem task_assignment_count : 
  let M := 3 -- number of males
  let F := 3 -- number of females
  let total_students := M + F
  -- The number of ways to choose 4 people from total_students such that task C has 2 people with at least one male and rest for tasks A and B.
  (C(M, 1) * C(F, 1) * A(4, 2) + C(M, 2) * A(4, 2)) = 144 :=
by
  sorry

end task_assignment_count_l384_384538


namespace solve_equation_l384_384920

theorem solve_equation (x : ℝ) :
  (1 / (x + 10) + 1 / (x + 8) = 1 / (x + 11) + 1 / (x + 7)) → x = -9 :=
by 
  sorry

end solve_equation_l384_384920


namespace num_divisors_90_l384_384042

theorem num_divisors_90 : (∀ (n : ℕ), n = 90 → (factors n).divisors.card = 12) :=
by {
  intro n,
  intro hn,
  sorry
}

end num_divisors_90_l384_384042


namespace rectangular_to_polar_l384_384640

noncomputable def polar_coordinates (x y : ℝ) : ℝ × ℝ :=
  let r := real.sqrt (x^2 + y^2) in
  let θ := real.pi - real.arctan (y / x) in
  (r, θ)

theorem rectangular_to_polar {x y : ℝ} (h₁ : x = -3) (h₂ : y = 4) :
  polar_coordinates x y = (5, real.pi - real.arctan (4 / 3)) :=
by
  rw [h₁, h₂]
  sorry

end rectangular_to_polar_l384_384640


namespace sara_sister_notebooks_l384_384135

theorem sara_sister_notebooks : 
  ∀ (original orderedPercent lost : ℤ), 
  original = 4 ∧ orderedPercent = 150 ∧ lost = 2 → 
  (original + (original * orderedPercent / 100) - lost) = 8 :=
by
  intros original orderedPercent lost
  intro h
  cases h with hOriginal hRest
  cases hRest with hOrderedPercent hLost
  rw [hOriginal, hOrderedPercent, hLost]
  norm_num
  sorry

end sara_sister_notebooks_l384_384135


namespace sam_letters_on_wednesday_l384_384910

/-- Sam's average letters per day. -/
def average_letters_per_day : ℕ := 5

/-- Number of days Sam wrote letters. -/
def number_of_days : ℕ := 2

/-- Letters Sam wrote on Tuesday. -/
def letters_on_tuesday : ℕ := 7

/-- Total letters Sam wrote in two days. -/
def total_letters : ℕ := average_letters_per_day * number_of_days

/-- Letters Sam wrote on Wednesday. -/
def letters_on_wednesday : ℕ := total_letters - letters_on_tuesday

theorem sam_letters_on_wednesday : letters_on_wednesday = 3 :=
by
  -- placeholder proof
  sorry

end sam_letters_on_wednesday_l384_384910


namespace max_value_m_l384_384783

def f (x: ℝ) : ℝ := x^3 - (9/2) * x^2 + 6 * x - 5

theorem max_value_m : ∃ m, (∀ x, f'(x) ≥ m) ∧ (∀ n, (∀ x, f'(x) ≥ n) → n ≤ m) ∧ m = - (3/4) := 
  sorry

end max_value_m_l384_384783


namespace minimize_sum_distances_l384_384365

open Real
open Set

variables {A B C O : EuclideanSpace ℝ (Fin 2)}

/-- For a triangle with all angles less than 120 degrees, the Fermat point minimizes
    the sum of distances to the vertices. For a triangle with one angle greater than or equal to 120 degrees,
    the vertex of this angle minimizes the sum of distances to the vertices. -/
theorem minimize_sum_distances (h₁: angle C A B < 120°) (h₂: angle A B C < 120°) (h₃: angle B C A < 120°) 
  (h₄: ¬ angle C A B < 120° ∨ ¬ angle A B C < 120° ∨ ¬ angle B C A < 120°) :
  (∀ O', inside_triangle ABC O' → dist O' A + dist O' B + dist O' C ≥ dist O A + dist O B + dist O C) :=
begin
  sorry
end

end minimize_sum_distances_l384_384365


namespace charlie_total_earnings_l384_384077

theorem charlie_total_earnings
  (wage : ℝ)
  (hours_week1 : ℝ)
  (hours_week2 : ℝ)
  (additional_earnings : ℝ)
  (constant_wage : ∀ t : ℝ, t ≠ 0 → wage = additional_earnings / t) :
  hours_week1 = 20 → hours_week2 = 30 → additional_earnings = 70 → constant_wage 10 → 
  (20 * wage + 30 * wage) = 350 := 
by
  intros hw1 hw2 ae cw
  rw [hw1, hw2] 
  have wage_eq : wage = 7 := by 
    calc 
      wage = additional_earnings / 10 := cw 10 (by norm_num)
      ... = 70 / 10 := by rw ae
      ... = 7 := by norm_num 
  calc 
    (20 * wage + 30 * wage) = 20 * 7 + 30 * 7 := by rw wage_eq
    ... = 350 := by norm_num

end charlie_total_earnings_l384_384077


namespace alpha_norm_six_l384_384467

theorem alpha_norm_six (α β : ℂ) (h1 : is_conj α β) (h2 : |α - β| = 6) (h3 : ∃ x : ℝ, α / β ^ 3 = x) : |α| = 6 := by
  sorry

end alpha_norm_six_l384_384467


namespace find_base_r_l384_384836

theorem find_base_r :
  ∀ (r : ℕ), (5 * r^2 + 3 * r + 1) + (4 * r^2 + 5 * r + 1) = r^3 + r^2 → r = 10 :=
by {
  intro r,
  sorry -- proof skipped
}

end find_base_r_l384_384836


namespace perfect_number_divisibility_l384_384602

noncomputable def isPerfectNumber (n : Nat) : Prop := 
  ∃ (m : Nat), m ∈ (setOf divisors n) ∧ (sumOfDivisors n = 2 * n)

theorem perfect_number_divisibility (p : Nat) (h1 : isPerfectNumber p) (h2 : p > 28) (h3 : 7 ∣ p) : 49 ∣ p := 
by 
  sorry

end perfect_number_divisibility_l384_384602


namespace number_of_ordered_quadruples_l384_384094

theorem number_of_ordered_quadruples :
  let n := ∑ (x : ℕ) in (finset.range 53).powerset_len 3, 1 in
  (n = 22100) ↔ 
  ∃ (x1 x2 x3 x4 : ℕ),
    (∀ i ∈ {x1, x2, x3, x4}, i % 2 = 1 ∧ i > 0) ∧
    (x1 + x2 + x3 + x4 = 102) :=
by sorry

end number_of_ordered_quadruples_l384_384094


namespace circle_equation_l384_384337

theorem circle_equation (x y : ℝ) :
    let center := (2, 1)
    let line := (3, 4, 5) -- The line is represented as (A, B, C) where Ax + By + C = 0
    let radius := abs (3 * 2 + 4 * 1 + 5) / sqrt (3^2 + 4^2)
    radius = 3 →
    (x - 2)^2 + (y - 1)^2 = 3^2 :=
by
  sorry

end circle_equation_l384_384337


namespace terminating_decimals_l384_384701

theorem terminating_decimals (n : ℤ) (h : 1 ≤ n ∧ n ≤ 180) :
  (∃ k : ℤ, k = 20 ∧ ∀ n, 1 ≤ n ∧ n ≤ 180 → (∃ m, m * 180 = n * (2^2 * 5))) := by
  sorry

end terminating_decimals_l384_384701


namespace John_investment_in_bank_A_l384_384456

noncomputable def investment_equation : Real := 
  let x := 1625
  let A := x * 1.04^3
  let B := (1950 - x) * 1.06^3
  A + B

theorem John_investment_in_bank_A : 
  ∃ x : ℝ, (0 ≤ x ∧ x ≤ 2000) ∧
  let A := x * 1.04^3 in
  let B := (1950 - x) * 1.06^3 in
  A + B = 2430 ∧ x = 1625 :=
by
  let x := 1625
  use x
  split
  { split; norm_num }
  split
  { calc x * 1.04^3 + (1950 - x) * 1.06^3
        = 1.124864 * 1625 + 1.191016 * (1950 - 1625)   : by norm_num
    ... = 1.124864 * 1625 + 1.191016 * 325
    ... = 1827.816 + 386.833 
    ... = 2430                                     
  sorry

end John_investment_in_bank_A_l384_384456


namespace circle_and_tangent_lines_pass_through_point_l384_384358

def circle_passing_through (r : ℝ) (Mx My : ℝ) : Prop :=
  Mx^2 + My^2 = r^2

def tangent_lines (r : ℝ) (Px Py : ℝ) (k1 k2 : ℝ) : Prop :=
  (Py - 2) * sqrt ((k1)^2 + 1) = 2 ∧ (Px - 3) * k1 = Py - 2 ∧
  (k2 = 0 ∨ k2 = 12/5)

theorem circle_and_tangent_lines_pass_through_point :
  (circle_passing_through 2 0 2) →
  (∃ t1 t2 : ℝ, tangent_lines 2 3 2 t1 t2) :=
by sorry

end circle_and_tangent_lines_pass_through_point_l384_384358


namespace terminating_decimal_fraction_count_l384_384693

theorem terminating_decimal_fraction_count :
  (finset.filter (λ n : ℕ, 1 ≤ n ∧ n ≤ 180 ∧ 90 ∣ n) (finset.range 181)).card = 20 :=
by sorry

end terminating_decimal_fraction_count_l384_384693


namespace angle_A_is_120_degrees_l384_384855

-- Given a trapezoid ABCD with AB and CD as parallel sides and angle relationships
variables (A D C B : ℝ) (trapezoid_ABCD : AB.parallel CD) (h1 : A = 2 * D) (h2 : C = 3 * B)

-- We need to prove that the measure of angle A is 120 degrees
theorem angle_A_is_120_degrees (h3 : A + D = 180) : A = 120 :=
by {
  sorry -- Here the proof would go
}

end angle_A_is_120_degrees_l384_384855


namespace flag_arrangements_l384_384553

theorem flag_arrangements (B R : ℕ) (M : ℕ) : 
  B = 12 ∧ R = 11 ∧ (∃ M, 
    M = (13 * Nat.choose 13 11 - 2 * Nat.choose 13 11)) →
  M % 1000 = 858 :=
by 
  intros h,
  obtain ⟨hB, hR, hM⟩ := h, 
  rw [hB, hR] at hM,
  simp [Nat.choose_eq_factorial_div_factorial, hM],
  sorry

end flag_arrangements_l384_384553


namespace terminating_decimal_count_l384_384753

theorem terminating_decimal_count : 
  (Finset.filter (λ n, n % 3 = 0) (Finset.range 181)).card = 60 := by
  sorry

end terminating_decimal_count_l384_384753


namespace inequality_holds_for_all_x_in_01_l384_384017

noncomputable def range_of_theta (θ : ℝ) : Prop :=
  ∃ k : ℤ, 2 * k * Real.pi + Real.pi / 12 < θ ∧ θ < 2 * k * Real.pi + 5 * Real.pi / 12

theorem inequality_holds_for_all_x_in_01 (θ : ℝ) : 
  (∀ x : ℝ, x ∈ Icc 0 1 → x^2 * Real.cos θ - x * (1 - x) + (1 - x)^2 * Real.sin θ > 0) →
  range_of_theta θ :=
by sorry

end inequality_holds_for_all_x_in_01_l384_384017


namespace solve_equation_l384_384926

def equation_holds (x : ℝ) : Prop := 
  (1 / (x + 10)) + (1 / (x + 8)) = (1 / (x + 11)) + (1 / (x + 7))

theorem solve_equation : equation_holds (-9) :=
by
  sorry

end solve_equation_l384_384926


namespace even_n_matrix_column_sums_equal_l384_384501

theorem even_n_matrix_column_sums_equal {n : ℕ} (h : n % 2 = 0) : 
  ∃ (M : matrix (fin n) (fin n) ℕ), (∀ j : fin n, ∑ i : fin n, M i j = ∑ i : fin n, M i 0) := 
sorry

end even_n_matrix_column_sums_equal_l384_384501


namespace line_and_circle_separate_l384_384805

theorem line_and_circle_separate
  (θ : ℝ) (hθ : ¬ ∃ k : ℤ, θ = k * Real.pi) :
  ¬ ∃ (x y : ℝ), (x^2 + y^2 = 1 / 2) ∧ (x * Real.cos θ + y - 1 = 0) :=
by
  sorry

end line_and_circle_separate_l384_384805


namespace value_of_x_l384_384570

-- Define the given condition
def x : ℝ := (2013^2 - 2013 - 1) / 2013

-- The goal is to prove the given condition equals the correct answer
theorem value_of_x : x = 2012 - (1 / 2013) :=
by
  sorry

end value_of_x_l384_384570


namespace minimal_sum_of_sequence_l384_384361

def a (n : ℕ) : ℤ := n^2 - 12 * n - 13

theorem minimal_sum_of_sequence :
  ∀ n : ℕ, (∑ k in Finset.range n, a k) = ∑ k in Finset.range 12, a k ∨ (∑ k in Finset.range n, a k) = ∑ k in Finset.range 13, a k :=
sorry

end minimal_sum_of_sequence_l384_384361


namespace limit_R_n_2_1_limit_n_R_n_2_2_l384_384883

noncomputable def X (i : ℕ) : Type := sorry -- Placeholder for the i.i.d. random variable X_i

-- Define the probability P(X_i > 0) = 1
def pos_prob : Prop := ∀ i : ℕ, Prob (X i > 0) = 1

-- Define the expectation of exp(-λ X_1)
def laplace_transform (λ : ℝ) : ℝ := sorry -- Placeholder for expectation E[exp(-λ X₁)]

-- Define R_n(l, α)
def R_n (n l : ℕ) (α : ℝ) : ℝ := sorry -- Placeholder for E[(X₁^l + ... + Xₙ^l) / (X₁ + ... + Xₙ)^α]

-- Define the expectations E[X₁] and E[X₁^2]
def E_X1 : ℝ := sorry -- Placeholder for E[X₁]
def E_X1_sq : ℝ := sorry -- Placeholder for E[X₁²]

-- The final goal to prove
theorem limit_R_n_2_1 (h_pos : pos_prob) (h_finite : E_X1_sq < ∞) :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, abs (R_n n 2 1 - E_X1_sq / E_X1) < ε :=
sorry

theorem limit_n_R_n_2_2 (h_pos : pos_prob) (h_finite : E_X1_sq < ∞) :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, abs (n * R_n n 2 2 - E_X1_sq / (E_X1^2)) < ε :=
sorry

end limit_R_n_2_1_limit_n_R_n_2_2_l384_384883


namespace no_such_complex_numbers_exist_l384_384384

open Complex

theorem no_such_complex_numbers_exist :
  ∀ (z1 z2 : ℂ), 
    (z2 - conj z2 ≠ 0 ∧ (z1 - conj z1) = 0 ∧ 
    (conj z2 + 6 = 2 / (z2 + 6)) ∧ 
    (z1 * z2^2 + z2 + 2 = 0)) → 
    False :=
by
  intros z1 z2 h
  cases h with h1 h2
  cases h2 with h3 h4
  simp at h4 h3
  sorry

end no_such_complex_numbers_exist_l384_384384


namespace final_digit_is_two_l384_384491

-- Define initial conditions
def initial_ones : ℕ := 10
def initial_twos : ℕ := 10

-- Define the possible moves and the parity properties
def erase_identical (ones twos : ℕ) : ℕ × ℕ :=
  if ones ≥ 2 then (ones - 2, twos + 1)
  else (ones, twos - 1) -- for the case where two twos are removed

def erase_different (ones twos : ℕ) : ℕ × ℕ :=
  (ones, twos - 1)

-- Theorem stating that the final digit must be a two
theorem final_digit_is_two : 
∀ (ones twos : ℕ), ones = initial_ones → twos = initial_twos → 
(∃ n, ones + twos = n ∧ n = 1 ∧ (ones % 2 = 0)) → 
(∃ n, ones + twos = n ∧ n = 0 ∧ twos = 1) := 
by
  intros ones twos h_ones h_twos condition
  -- Constructing the proof should be done here
  sorry

end final_digit_is_two_l384_384491


namespace rational_k_quadratic_solution_count_l384_384536

theorem rational_k_quadratic_solution_count (N : ℕ) :
  (N = 98) ↔ 
  (∃ (k : ℚ) (x : ℤ), |k| < 500 ∧ (3 * x^2 + k * x + 7 = 0)) :=
sorry

end rational_k_quadratic_solution_count_l384_384536


namespace A_n_is_integer_l384_384378

-- Defining the problem conditions
variables (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_gt : a > b)

def θ (a b : ℕ) : ℝ := Real.arcsin ((2 * a * b) / (a ^ 2 + b ^ 2))

def A_n (a b : ℕ) (n : ℕ) : ℝ := (a ^ 2 + b ^ 2) ^ n * Real.sin (n * θ a b)

theorem A_n_is_integer (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_gt : a > b) : 
  ∀ n : ℕ, ∃ (k : ℤ), A_n a b n = k :=
by
  sorry

end A_n_is_integer_l384_384378


namespace real_root_sum_eq_l384_384187

-- Define the condition as the absolute value equation and the roots as per the solution
noncomputable def root_sum : ℝ :=
  (if h : (-3 ≤ -6/5 ∧ -6/5 ≤ 1) then -6/5 else 0) +
  (if h : (2 < 1 + sqrt 97 / 4) then (1 + sqrt 97) / 4 else 0)

theorem real_root_sum_eq : (root_sum = (5 * sqrt 97 - 19) / 20) := sorry

end real_root_sum_eq_l384_384187


namespace exists_color_removal_connected_l384_384543

noncomputable theory

open_locale classical

def K20_colored := simple_graph (fin 20) -- Define the complete graph K20

-- Define the complete graph with edges being one of five colors
axiom colored_edges : K20_colored → fin 5

theorem exists_color_removal_connected :
  ∃ c : fin 5, ∀ e, colored_edges e ≠ c → (K20_colored - {e}) .conn :=
sorry

end exists_color_removal_connected_l384_384543


namespace weekend_rental_rate_per_day_l384_384932

theorem weekend_rental_rate_per_day :
  ∀ (persons : ℕ) (days : ℕ) (weekday_cost_per_day : ℝ) (total_payment_per_person : ℝ)
  (total_days : ℕ) (weekdays : ℕ) (weekends : ℕ), 
  persons = 6 ∧
  days = 4 ∧
  weekday_cost_per_day = 420 ∧
  total_payment_per_person = 320 ∧
  total_days = 4 ∧
  weekdays = 2 ∧
  weekends = 2 →
  let total_weekday_cost := weekdays * weekday_cost_per_day in
  let total_payment := persons * total_payment_per_person in
  let total_weekend_cost := total_payment - total_weekday_cost in
  let weekend_cost_per_day := total_weekend_cost / weekends in
  weekend_cost_per_day = 540 := by
sorry

end weekend_rental_rate_per_day_l384_384932


namespace total_pizza_slices_l384_384895

theorem total_pizza_slices (num_pizzas : ℕ) (slices_per_pizza : ℕ) : num_pizzas = 2 → slices_per_pizza = 8 → num_pizzas * slices_per_pizza = 16 := by
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

end total_pizza_slices_l384_384895


namespace problem1_problem2_l384_384248

-- Define the first problem
theorem problem1 (x : ℝ) : (x - 2) ^ 2 = 2 * x - 4 ↔ (x = 2 ∨ x = 4) := 
by 
  sorry

-- Define the second problem using completing the square method
theorem problem2 (x : ℝ) : x ^ 2 - 4 * x - 1 = 0 ↔ (x = 2 + Real.sqrt 5 ∨ x = 2 - Real.sqrt 5) := 
by 
  sorry

end problem1_problem2_l384_384248


namespace distance_between_A_and_B_l384_384496

theorem distance_between_A_and_B (vA vB vC : ℝ) (dA dB dC dCD : ℝ) :
  vA = 3 * vC →
  vA = 1.5 * vB →
  dCD = 12 →
  dA = 50 + 3 * dB →
  dB = (50 + 3 * dBD) * (2/3) →
  dA - dB = dCD →
  dA + dB = dA + 3 * dBD - dCD →
  dA + 3 * dBD + 50 = 130 :=
begin
  -- sorry is included to skip the proof as per the instructions
  sorry
end

end distance_between_A_and_B_l384_384496


namespace graph_symmetric_about_point_l384_384948

noncomputable def tan_shifted_symmetric : Prop :=
  ∀ x : ℝ, tan (2 * (x + π / 6) + π / 6) = g(x)

theorem graph_symmetric_about_point :
  ∀ x : ℝ, tan (2 * x + π / 2) = g(x) → g(π / 4) = 0 :=
sorry

end graph_symmetric_about_point_l384_384948


namespace largest_prime_factor_1023_l384_384566

theorem largest_prime_factor_1023 : 
  (∃ p, p ∣ 1023 ∧ nat.prime p) ∧ 
  (∀ q, q ∣ 1023 ∧ nat.prime q → q ≤ 31) :=
by
  let p3 := 3
  let p11 := 11
  let p31 := 31
  have h1 : 1023 = p3 * 341 := rfl
  have h2 : 341 = p11 * p31 := rfl
  have h3 : nat.prime p3 := by sorry
  have h4 : nat.prime p11 := by sorry
  have h5 : nat.prime p31 := by sorry
  -- Proof of the theorem using these conditions
  sorry

end largest_prime_factor_1023_l384_384566


namespace total_seashells_l384_384132

-- Conditions
def sam_seashells : Nat := 18
def mary_seashells : Nat := 47

-- Theorem stating the question and the final answer
theorem total_seashells : sam_seashells + mary_seashells = 65 :=
by
  sorry

end total_seashells_l384_384132


namespace terminating_decimals_count_l384_384717

theorem terminating_decimals_count :
  ∃ (count : ℕ), count = 60 ∧
  ∀ (n : ℕ), 1 ≤ n ∧ n ≤ 180 →
  (∃ m : ℕ, m * 180 = n * (2^2 * 5)) → 60 = count := sorry

end terminating_decimals_count_l384_384717


namespace lambda_value_l384_384809

theorem lambda_value (a : ℕ → ℤ) (λ : ℤ) :
  (∀ n : ℕ, 0 < n → a (n + 1) = 2 * a n + 2^n - 1) →
  (∀ n : ℕ, 0 < n → (a (n + 1) + λ) / (2^(n + 1)) - (a n + λ) / (2^n) = (a 1 + λ) / (2^1) - (a 0 + λ) / (2^0)) →
  λ = -1 := by
  sorry

end lambda_value_l384_384809


namespace candy_blue_pieces_l384_384968

theorem candy_blue_pieces (total_pieces : ℕ) (red_pieces : ℕ) (blue_pieces : ℕ) 
  (h1 : total_pieces = 11567) 
  (h2 : red_pieces = 792) 
  (h3 : blue_pieces = total_pieces - red_pieces) : 
  blue_pieces = 10775 := 
by 
  rw [h1, h2] at h3 
  exact h3

#eval candy_blue_pieces 11567 792 (11567 - 792) (by rfl) (by rfl) (by simp)

end candy_blue_pieces_l384_384968


namespace trigonometric_identity_l384_384404

theorem trigonometric_identity (x : ℝ) (h : Real.sin x + Real.cos x + Real.tan x + Real.cot x + Real.sec x + Real.csc x = 9) : Real.cos (2 * x) = 1 :=
by
  sorry

end trigonometric_identity_l384_384404


namespace triangle_third_side_lengths_l384_384828

noncomputable def valid_third_side_lengths :=
  {n : ℤ | 2 < n ∧ n < 18}

theorem triangle_third_side_lengths :
  valid_third_side_lengths = {n : ℤ | 2 < n ∧ n < 18} →
  valid_third_side_lengths.card = 15 := 
by
  sorry

end triangle_third_side_lengths_l384_384828


namespace seashells_count_l384_384134

theorem seashells_count : 18 + 47 = 65 := by
  sorry

end seashells_count_l384_384134


namespace john_payment_least_days_l384_384455

theorem john_payment_least_days (initial_borrowed : ℕ) (daily_interest_rate : ℝ) 
(h1 : initial_borrowed = 50) (h2 : daily_interest_rate = 0.1) 
: ∃ (x : ℕ), 50 + 5 * x ≥ 100 ∧ ∀ (y : ℕ), 50 + 5 * y ≥ 100 → y ≥ 10 := 
by
  use 10
  split
  . exact sorry  -- left proof part: showing 50 + 5 * 10 ≥ 100
  . exact sorry  -- right proof part: showing any y that satisfies 50 + 5 * y ≥ 100 must be ≥ 10

end john_payment_least_days_l384_384455


namespace initial_thickness_after_folds_l384_384600

theorem initial_thickness_after_folds (final_thickness : ℝ) (n : ℕ) (folds : n = 50) (thickness_after_folds : final_thickness = 1) :
  ∃ (initial_thickness : ℝ), initial_thickness = final_thickness / (2 ^ n) :=
by
  use final_thickness / (2 ^ n)
  sorry

end initial_thickness_after_folds_l384_384600


namespace terminating_decimals_l384_384710

theorem terminating_decimals (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 180) :
  (∃ k : ℕ, k = 20 ∧ ∀ m : ℕ, 1 ≤ m ∧ m ≤ 180 → (decimal_terminates (m / 180) ↔ m % 9 = 0)) :=
by {
  sorry
}

def decimal_terminates (x : ℚ) : Prop :=
  sorry

end terminating_decimals_l384_384710


namespace translate_line_downward_l384_384974

theorem translate_line_downward (x y : ℝ) : (y = 4 * x) → (∃ y', y' = 4 * x - 5) :=
by
    intro h
    use y - 5
    rw h
    norm_num
    done

end translate_line_downward_l384_384974


namespace increasing_decreasing_intervals_range_of_a_increasing_l384_384028

section
variable (a : ℝ)

def quadratic_function (x : ℝ) := x^2 + 2 * a * x + 3

-- Define the intervals of increase or decrease for the case a = -1
theorem increasing_decreasing_intervals :
  let f := quadratic_function (-1)
  ∀ x, x ∈ Icc (-4 : ℝ) 1 → f x ≤ f 1 ∧
        x ∈ Ico 1 6 → f 1 ≤ f x :=
  sorry

-- Define the range of a for which the function is monotonically increasing on [-4, 6]
theorem range_of_a_increasing :
  {a : ℝ | 
   ∀ x y, x ∈ Icc (-4 : ℝ) 6 → y ∈ Icc (-4 : ℝ) 6 → x ≤ y → 
   quadratic_function a x ≤ quadratic_function a y} = 
   {a | a ≤ -6 ∨ a ≥ 4} :=
  sorry
end

end increasing_decreasing_intervals_range_of_a_increasing_l384_384028


namespace inequality_problem_l384_384764

variable {R : Type} [Preorder R] [Add R] [Mul R] [Div R] [Sub R] [Zero R]

theorem inequality_problem 
  (k : R) (a b c : R)
  (hk : 1 < k)
  (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c)
  (h1 : a ≤ k * c)
  (h2 : b ≤ k * c)
  (h3 : a * b ≤ c * c) :
  a + b ≤ (k + k⁻¹) * c :=
sorry

end inequality_problem_l384_384764


namespace ratio_of_diagonal_intersection_points_l384_384097

theorem ratio_of_diagonal_intersection_points
  (A B C D P : Type) [MetricSpace A]
  (on_circle : ∀ {X Y : A}, X ≠ Y → Metric.distance X Y ≠ 0 ∧ Metric.distance X Y = 2)
  (AB : Metric.distance A B = 3)
  (BC : Metric.distance B C = 5)
  (CD : Metric.distance C D = 6)
  (DA : Metric.distance D A = 4)
  (intersect_P : P = intersection (line A C) (line B D))
  : AP / CP = 2 / 5 :=
by sorry

end ratio_of_diagonal_intersection_points_l384_384097


namespace point_P_traces_ellipse_l384_384396

-- Definitions based on the conditions
variables (A B C D P : Point)
variables (AB_CD_parallel : is_parallel AB CD)
variables (AB_CD_inscribed : is_inscribed_trapezoid A B C D)
variables (P_on_CD: is_on_segment P CD)

-- Theorem statement
theorem point_P_traces_ellipse (AB_CD_parallel : is_parallel AB CD) 
  (AB_CD_inscribed : is_inscribed_trapezoid A B C D)
  (P_on_CD : is_on_segment P CD) : traces_ellipse P AB CD := 
sorry

end point_P_traces_ellipse_l384_384396


namespace prob_article_error_free_l384_384584

theorem prob_article_error_free (p_catch : ℚ) (p_miss : ℚ) (days : ℕ) : p_catch = 2/3 ∧ p_miss = 1 - p_catch ∧ days = 3 →
  (1 - (p_miss^days)) * (1 - (p_miss^(days-1))) * (1 - (p_miss^(days-2))) = 416/729 :=
begin
  intro h,
  have h1 : p_catch = 2/3 := h.1,
  have h2 : p_miss = 1 - p_catch := h.2.1,
  have h3 : days = 3 := h.2.2,
  sorry,
end

end prob_article_error_free_l384_384584


namespace log_func_passes_through_fixed_point_l384_384945

theorem log_func_passes_through_fixed_point (a : ℝ) (ha1 : 0 < a) (ha2 : a ≠ 1) : 
  ∃ (x y : ℝ), x = 2 ∧ y = 2 ∧ (∀ x : ℝ, f x = 2 + log a (x - 1)) :=
by
  set f := λ x, 2 + log a (x - 1)
  have key_point : f 2 = 2 := by
    simp [f]
    rw [log a 1, add_zero]
  exact ⟨2, 2, rfl, key_point⟩

end log_func_passes_through_fixed_point_l384_384945


namespace prism_lateral_area_l384_384769

-- Definitions for conditions
def regular_triangular_base : Prop := sorry
def lateral_edges_perpendicular : Prop := sorry
def sphere_volume : ℝ := (4/3) * π

-- Main problem statement to prove
theorem prism_lateral_area (prism : Type) (r : ℝ) (s : ℝ) (h : ℝ)
  (h_prism : regular_triangular_base ∧ lateral_edges_perpendicular ∧ sphere_volume = (4/3) * π ∧ r = 1 ∧ s = 2 * sqrt 3 ∧ h = 2) :
  3 * s * h = 12 * sqrt 3 :=
sorry

end prism_lateral_area_l384_384769


namespace cosine_value_of_angle_l384_384778

-- Define the points and vectors
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (2, -1)
def a : ℝ × ℝ := (-2, 1)
def b : ℝ × ℝ := (1, 1)

-- Define the vector AB
def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

-- Define the vector a - b
def a_minus_b : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)

-- Compute the cosine of the angle
def cosine_angle (u v : ℝ × ℝ) : ℝ :=
  let dot_product := u.1 * v.1 + u.2 * v.2
  let magnitude_u := Real.sqrt (u.1 * u.1 + u.2 * u.2)
  let magnitude_v := Real.sqrt (v.1 * v.1 + v.2 * v.2)
  dot_product / (magnitude_u * magnitude_v)

theorem cosine_value_of_angle :
  cosine_angle AB a_minus_b = - Real.sqrt 5 / 5 := sorry

end cosine_value_of_angle_l384_384778


namespace prob_same_color_eq_19_div_39_l384_384228

/-- Definition of initial conditions --/
def total_balls : ℕ := 5 + 8
def green_balls : ℕ := 5
def white_balls : ℕ := 8

/-- Helper functions to calculate probabilities --/
def prob_green_first : ℚ := green_balls / total_balls
def prob_green_second : ℚ := (green_balls - 1) / (total_balls - 1)
def prob_both_green : ℚ := prob_green_first * prob_green_second

def prob_white_first : ℚ := white_balls / total_balls
def prob_white_second : ℚ := (white_balls - 1) / (total_balls - 1)
def prob_both_white : ℚ := prob_white_first * prob_white_second

def prob_same_color : ℚ := prob_both_green + prob_both_white

/-- The main proof statement --/
theorem prob_same_color_eq_19_div_39 :
  prob_same_color = 19 / 39 :=
by
  -- This is where the proof would be constructed.
  sorry

end prob_same_color_eq_19_div_39_l384_384228


namespace angle_B_value_range_a_l384_384448

-- Part 1
theorem angle_B_value (a b : ℝ) (A B : ℝ) (hb : b = 10) (hA : A = π / 6) (ha : a = 5) :
  B = π / 2 :=
by
  sorry

-- Part 2
theorem range_a (a b : ℝ) (A B : ℝ) (hb : b = 10) (hA : A = π / 6) (hB_cond : ∀ B, 0 < sin B ∧ sin B < 1) :
  5 < a ∧ a < 10 :=
by
  sorry

end angle_B_value_range_a_l384_384448


namespace perimeter_of_ABCDE_l384_384073

noncomputable def point := (ℝ × ℝ)

def A : point := (0, 10)
def B : point := (10, 10)
def C : point := (10, 0)
def D : point := (0, 5)
def E : point := (10, 5)

def length (p1 p2 : point) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def perimeter_ABCDE : ℝ :=
  length A B + length B C + length C D + length D E + length E A

theorem perimeter_of_ABCDE : perimeter_ABCDE = 30 + 5 * real.sqrt (2 - real.sqrt 2) :=
by sorry

end perimeter_of_ABCDE_l384_384073


namespace horse_speed_proof_l384_384936

noncomputable def square_side_length (area : ℝ) : ℝ :=
  real.sqrt area

noncomputable def square_perimeter (side : ℝ) : ℝ :=
  4 * side

noncomputable def horse_speed (perimeter : ℝ) (time : ℝ) : ℝ :=
  perimeter / time

theorem horse_speed_proof :
  let area := 1600
  let time := 10
  let side := square_side_length area
  let perimeter := square_perimeter side
  horse_speed perimeter time = 16 :=
by
  sorry

end horse_speed_proof_l384_384936


namespace no_tiling_possible_l384_384582

-- Definitions based on conditions
def board := fin 100 × fin 100
def removed_square (sq : board) : set board := {b | b ≠ sq}

-- Problem statement formalized
theorem no_tiling_possible (sq : board) :
  ¬ ∃ (t : set (set board)),
    (∀ triangle ∈ t, is_triangle triangle) ∧
    (∀ triangle ∈ t, hypotenuse_on_sides triangle) ∧
    (∀ triangle ∈ t, legs_on_diagonals triangle) ∧
    (disjoint_union t = removed_square sq ∧
    (∀ t₁ t₂ ∈ t, t₁ ≠ t₂ → disjoint t₁ t₂)) :=
sorry

end no_tiling_possible_l384_384582


namespace part1_min_value_and_period_part2_triangle_sides_l384_384801

noncomputable def f (x : ℝ) : ℝ :=
  (sqrt 3) * sin x * cos x - (1 / 2) * cos (2 * x) - 1

theorem part1_min_value_and_period : 
    (∃ x : ℝ, f x = -2) ∧ 
    (∃ p : ℝ, p > 0 ∧ ∀ x : ℝ, f (x + p) = f x :=) := 
sorry

variables {A B C a b c: ℝ}

def m := (1, sin A)
def n := (2, sin B)

theorem part2_triangle_sides 
  (hc : c = 3) 
  (hfC : f C = 0)
  (hC : C = π / 3)
  (hmn_collinear : ∃ k : ℝ, k * m = n) :
  a = sqrt 3 ∧ b = 2 * sqrt 3 :=
sorry

end part1_min_value_and_period_part2_triangle_sides_l384_384801


namespace terminating_decimals_count_l384_384731

theorem terminating_decimals_count : 
  let N := 180 in
  let count_terminating_decimals := (List.range' 1 N).filter (λ n, n % 3 = 0) in
  count_terminating_decimals.length = 60 :=
by
  let N := 180
  let count_terminating_decimals := (List.range' 1 N).filter (λ n, n % 3 = 0)
  sorry

end terminating_decimals_count_l384_384731


namespace toy_spending_ratio_l384_384450

theorem toy_spending_ratio :
  ∃ T : ℝ, 204 - T > 0 ∧ 51 = (204 - T) / 2 ∧ (T / 204) = 1 / 2 :=
by
  sorry

end toy_spending_ratio_l384_384450


namespace swimming_time_l384_384598

theorem swimming_time (c t : ℝ) 
  (h1 : 10.5 + c ≠ 0)
  (h2 : 10.5 - c ≠ 0)
  (h3 : t = 45 / (10.5 + c))
  (h4 : t = 18 / (10.5 - c)) :
  t = 3 := 
by
  sorry

end swimming_time_l384_384598


namespace sum_first_89_terms_eq_100_l384_384078

noncomputable def sum_of_sequence (n : ℕ) : ℕ :=
  if n % 8 = 0 then 9 * (n / 8) else
    let quotient := n / 8 in
    let remainder := n % 8 in
    9 * quotient + 
    match remainder with
    | 1 => 1
    | 2 => 3
    | 3 => 3
    | 4 => 5
    | 5 => 7
    | 6 => 8
    | 7 => 9
    | _ => 0
    end

theorem sum_first_89_terms_eq_100 : sum_of_sequence 89 = 100 :=
  sorry

end sum_first_89_terms_eq_100_l384_384078


namespace eliminate_y_substitution_l384_384757

theorem eliminate_y_substitution (x y : ℝ) (h1 : y = x - 5) (h2 : 3 * x - y = 8) : 3 * x - x + 5 = 8 := 
by
  sorry

end eliminate_y_substitution_l384_384757


namespace terminating_decimals_l384_384695

theorem terminating_decimals (n : ℤ) (h : 1 ≤ n ∧ n ≤ 180) :
  (∃ k : ℤ, k = 20 ∧ ∀ n, 1 ≤ n ∧ n ≤ 180 → (∃ m, m * 180 = n * (2^2 * 5))) := by
  sorry

end terminating_decimals_l384_384695


namespace car_owners_without_motorcycle_or_bicycle_l384_384967

noncomputable def total_adults := 500
noncomputable def car_owners := 400
noncomputable def motorcycle_owners := 200
noncomputable def bicycle_owners := 150
noncomputable def car_motorcycle_owners := 100
noncomputable def motorcycle_bicycle_owners := 50
noncomputable def car_bicycle_owners := 30

theorem car_owners_without_motorcycle_or_bicycle :
  car_owners - car_motorcycle_owners - car_bicycle_owners = 270 := by
  sorry

end car_owners_without_motorcycle_or_bicycle_l384_384967


namespace area_of_triangle_OAB_is_1_l384_384031

noncomputable def vector_a : ℝ × ℝ := (-1/2, real.sqrt (3) / 2)
variables (b : ℝ × ℝ) (O A B : ℝ × ℝ)

theorem area_of_triangle_OAB_is_1
  (h1 : A = vector_a - b)
  (h2 : B = vector_a + b)
  (h3 : ∀ a b : ℝ × ℝ, a = vector_a → b ≠ 0 → (a - b) ∧ (a + b) = a ∧ b) :
  let OA := A - O,
      OB := B - O in
  ∥OA∥ = real.sqrt 2 ∧ ∥OB∥ = real.sqrt 2 ∧ ∀ a b : ℝ × ℝ, (a - b).orthogonal (a + b) → 
  (1 / 2) * ∥OA∥ * ∥OB∥ = 1 :=
begin
  sorry
end

end area_of_triangle_OAB_is_1_l384_384031


namespace integer_satisfying_conditions_l384_384213

theorem integer_satisfying_conditions :
  {a : ℤ | 1 ≤ a ∧ a ≤ 105 ∧ 35 ∣ (a^3 - 1)} = {1, 11, 16, 36, 46, 51, 71, 81, 86} :=
by
  sorry

end integer_satisfying_conditions_l384_384213


namespace area_triangle_not_less_four_ninths_l384_384056

theorem area_triangle_not_less_four_ninths 
  (A B C P E F : Type) 
  (h₁ : ∀ (A B C : ℝ), S (triangle A B C) = 1) 
  (hP : P ∈ segment B C) 
  (hPE : parallel PE BA) 
  (hPF : parallel PF CA) :
  S (triangle B P F) ≥ 4 / 9 ∨ S (triangle P C E) ≥ 4 / 9 ∨ S (quadrilateral P E A F) ≥ 4 / 9 :=
sorry

end area_triangle_not_less_four_ninths_l384_384056


namespace steve_take_home_pay_l384_384930

def salary : ℕ := 40000
def tax_deduction_rate : ℝ := 0.20
def healthcare_deduction_rate : ℝ := 0.10
def union_dues : ℕ := 800

theorem steve_take_home_pay : 
  let total_deductions : ℝ := (salary * tax_deduction_rate) + (salary * healthcare_deduction_rate) + union_dues in
  (salary : ℝ) - total_deductions = 27200 :=
sorry

end steve_take_home_pay_l384_384930


namespace plane_through_point_at_distance_to_line_l384_384274

variables {Point Line : Type} [Geometry Point Line]

def given_point (A : Point) : Prop := True
def given_line (a : Line) : Prop := True
def given_distance (c : ℝ) : Prop := c > 0

theorem plane_through_point_at_distance_to_line (A : Point) (a : Line) (c : ℝ)
  (hA : given_point A) (ha : given_line a) (hc : given_distance c) :
  ∃ (P : Plane), P.contains A ∧ P.is_at_distance c a :=
sorry

end plane_through_point_at_distance_to_line_l384_384274


namespace even_sine_function_phi_l384_384054

open Real

theorem even_sine_function_phi (φ : ℝ) (h : ∀ x, sin(2 * x + φ) = sin(-2 * x + φ)) : 
  φ = π / 2 :=
by
  sorry

end even_sine_function_phi_l384_384054


namespace terminating_decimals_count_l384_384720

theorem terminating_decimals_count :
  ∃ (count : ℕ), count = 60 ∧
  ∀ (n : ℕ), 1 ≤ n ∧ n ≤ 180 →
  (∃ m : ℕ, m * 180 = n * (2^2 * 5)) → 60 = count := sorry

end terminating_decimals_count_l384_384720


namespace proof_AB_eq_VW_l384_384884

variables {A B C D X Y Z V W : Type*}

-- Assume definitions and conditions
variables [Triangle A B C] (D_on_BC : Point D ∈ LineSegment B C)
variables (line_through_D : ∃ (lx ly : Line), lx ∋ D ∧ lx ∋ X ∧ lx ∋ B ∧ ly ∋ D ∧ ly ∋ Y ∧ ly ∋ C)
variables (circumcircle_BXD : ∃ (circle_BXD : Circle), ∀ (P : Point), P ∈ circle_BXD ↔ ∃ (angle : Angle), ∠BXD = angle ∧ P ∉ B)
variables (circumcircle_ABC : Circle ABC)

-- The circumcircle of triangle BXD intersects the circumcircle ω of triangle ABC again at point Z ≠ B
variables (circumcircle_intersection : ∀ P ∈ circumcircle_BXD, P ≠ B → P ∈ circumcircle_ABC → P = Z)

-- Lines ZD and ZY intersect ω again at V and W, respectively
variables (intersection_D : ∀ P ∈ circumcircle_ABC, line_through Z D P → P = V)
variables (intersection_Y : ∀ P ∈ circumcircle_ABC, line_through Z Y P → P = W)

-- The final proof
theorem proof_AB_eq_VW
  (h1 : Triangle A B C)
  (h2 : Point D ∈ LineSegment B C)
  (h3 : ∃ (lx ly : Line), lx ∋ D ∧ lx ∋ X ∧ lx ∋ B ∧ ly ∋ D ∧ ly ∋ Y ∧ ly ∋ C)
  (h4 : ∃ (circle_BXD : Circle), ∀ (P : Point), P ∈ circle_BXD ↔ ∃ (angle : Angle), ∠BXD = angle ∧ P ∉ B)
  (h5 : Circle ABC)
  (h6 : ∀ P ∈ circle_BXD, P ≠ B → P ∈ Circle ABC → P = Z)
  (h7 : ∀ P ∈ Circle ABC, Line_through Z D P → P = V)
  (h8 : ∀ P ∈ Circle ABC, Line_through Z Y P → P = W) :
  AB = VW :=
by
  sorry

end proof_AB_eq_VW_l384_384884


namespace num_terminating_decimals_l384_384683

theorem num_terminating_decimals (n : ℕ) (h1 : 1 ≤ n ∧ n ≤ 180) :
  (∃ k : ℕ, k = 20) := by
  have h2 : ∀ n, (1 ≤ n ∧ n ≤ 180) → (∃ k, k = (180 / 9)) := sorry
  have h3 : 180 / 9 = 20 := by norm_num
  use 20
  exact h3

end num_terminating_decimals_l384_384683


namespace intersection_eq_l384_384373

open Set

variable {α : Type*}

def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

theorem intersection_eq : M ∩ N = {2, 3} := by
  apply Set.ext
  intro x
  simp [M, N]
  sorry

end intersection_eq_l384_384373


namespace minimum_coins_paid_l384_384010

theorem minimum_coins_paid (k : ℕ) (h : k > 0) : 
  let total_days_paid := ∑ i in Finset.range (2 * k), (i + 1) * (2 * k - i) / 2
  ∃ f : ℕ → ℤ, (∀ k > 0, f k = (4 * k^3 + k^2 - k) / 2) ∧
    total_days_paid = f k := sorry

end minimum_coins_paid_l384_384010


namespace retailer_profit_percent_l384_384997

noncomputable def cost_price (purchase_price overhead_expenses : ℝ) : ℝ :=
  purchase_price + overhead_expenses

noncomputable def profit (selling_price cost_price : ℝ) : ℝ :=
  selling_price - cost_price

noncomputable def profit_percent (profit cost_price : ℝ) : ℝ :=
  (profit / cost_price) * 100

theorem retailer_profit_percent (purchase_price overhead_expenses selling_price : ℝ)
  (h_purchase_price : purchase_price = 225)
  (h_overhead_expenses : overhead_expenses = 20)
  (h_selling_price : selling_price = 300) :
  profit_percent (profit selling_price (cost_price purchase_price overhead_expenses)) (cost_price purchase_price overhead_expenses) ≈ 22.45 := 
by
  -- While the detailed proof is omitted as per instruction, we acknowledge the computation may lead to a numerical approximation.
  sorry

end retailer_profit_percent_l384_384997


namespace orthocentric_tetrahedron_l384_384439

theorem orthocentric_tetrahedron (ABCD : Type) (A B C D : ABCD) 
  (h a b c : ℝ) 
  (is_orthocentric : orthocentric_tetrahedron ABCD)
  (right_angle_ADC : ∠ADC = π / 2)
  (altitude : height D (plane A B C) = h) 
  (a_def : dist D A = a) 
  (b_def : dist D B = b) 
  (c_def : dist D C = c) :
  1 / h^2 = 1 / a^2 + 1 / b^2 + 1 / c^2 := by
  sorry

end orthocentric_tetrahedron_l384_384439


namespace terminating_decimals_count_l384_384726

theorem terminating_decimals_count : 
  let N := 180 in
  let count_terminating_decimals := (List.range' 1 N).filter (λ n, n % 3 = 0) in
  count_terminating_decimals.length = 60 :=
by
  let N := 180
  let count_terminating_decimals := (List.range' 1 N).filter (λ n, n % 3 = 0)
  sorry

end terminating_decimals_count_l384_384726


namespace slope_tangent_a_eq_1_range_of_a_min_value_range_of_a_monotonic_l384_384023

def f (a x : ℝ) : ℝ := a * x^2 - (a + 2) * x + Real.log x

def f_prime (a x : ℝ) : ℝ := 2 * x - 3 + 1 / x

-- Slope of the tangent line at (1, f(1))
theorem slope_tangent_a_eq_1 (hf : f 1 1 = 1^2 - 3 * 1 + Real.log 1) :
  f_prime 1 1 = 0 :=
by sorry

-- Range of a for minimum value -2 on the interval [1, e]
theorem range_of_a_min_value (a : ℝ) (hₐ : a > 0) 
    (hfₐ : ∀ x ∈ Icc (1 : ℝ) Real.exp, f a x ≥ -2) :
  1 ≤ a :=
by sorry

-- f(x₁) + 2x₁ < f(x₂) + 2x₂ implies range of a
theorem range_of_a_monotonic (a : ℝ)
    (hf_cond : ∀ x1 x2 : ℝ, x1 < x2 → x1 > 0 → x2 > 0 → f a x1 + 2 * x1 < f a x2 + 2 * x2) :
  0 ≤ a ∧ a ≤ 8 :=
by sorry

end slope_tangent_a_eq_1_range_of_a_min_value_range_of_a_monotonic_l384_384023


namespace select_50_boxes_l384_384243

theorem select_50_boxes (n : ℕ) (apples oranges : Fin n → ℕ) (h_n : n = 99) :
  ∃ (S : Finset (Fin n)), S.card = 50 ∧
    (∑ i in S, apples i) ≥ (∑ i, apples i) / 2 ∧
    (∑ i in S, oranges i) ≥ (∑ i, oranges i) / 2 := by
  sorry

end select_50_boxes_l384_384243


namespace min_value_of_xy_ratio_l384_384142

theorem min_value_of_xy_ratio :
  ∃ t : ℝ,
    (t = 2 ∨
    t = ((-1 + Real.sqrt 217) / 12) ∨
    t = ((-1 - Real.sqrt 217) / 12)) ∧
    min (min 2 ((-1 + Real.sqrt 217) / 12)) ((-1 - Real.sqrt 217) / 12) = -1.31 :=
sorry

end min_value_of_xy_ratio_l384_384142


namespace hyperbola_asymptotes_l384_384393

def point_on_hyperbola (a b x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

def hyperbola_asymptotes_eq (a b : ℝ) : Prop :=
  ∀ (y x : ℝ), y = (b / a) * x ∨ y = -(b / a) * x ↔ y = x ∨ y = -x

def condition_1 (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0

def condition_2 (F1 F2 P1 P2 : ℝ) (d1 d2 : ℝ) : Prop :=
  |F1 - F2|^2 = 16 * d1 * d2

noncomputable def distances (a b x0 y0 : ℝ) :
  ℝ × ℝ :=
  let c := (a^2 + b^2).sqrt in
  let d1 := |b * x0 - a * y0| / c in
  let d2 := |b * x0 + a * y0| / c in
  (d1, d2)

theorem hyperbola_asymptotes (a b F1 F2 x0 y0 : ℝ) :
  condition_1 a b →
  point_on_hyperbola a b x0 y0 →
  let (d1, d2) := distances a b x0 y0 in
  condition_2 F1 F2 d1 d2 →
  hyperbola_asymptotes_eq a b :=
by sorry

end hyperbola_asymptotes_l384_384393


namespace pool_depth_l384_384935

variable (drain_rate : ℕ) (width : ℕ) (length : ℕ) (capacity_ratio : ℚ) (drain_time : ℕ) 
variable (drain_rate_eq : drain_rate = 60) 
variable (width_eq : width = 40) 
variable (length_eq : length = 150) 
variable (capacity_ratio_eq : capacity_ratio = 0.8) 
variable (drain_time_eq : drain_time = 800)

theorem pool_depth :
  let volume_drained := drain_rate * drain_time,
      total_capacity := volume_drained / capacity_ratio,
      depth := total_capacity / (length * width) in
  depth = 10 :=
by
  sorry

end pool_depth_l384_384935


namespace number_of_divisors_l384_384037

-- Defining the given number and its prime factorization as a condition.
def given_number : ℕ := 90

-- Defining the prime factorization.
def prime_factorization (n : ℕ) : List (ℕ × ℕ) :=
  if n = 90 then [(2, 1), (3, 2), (5, 1)] else []

-- The statement to prove that the number of positive divisors of 90 is 12.
theorem number_of_divisors (n : ℕ) (pf : List (ℕ × ℕ)) :
  n = 90 → pf = [(2, 1), (3, 2), (5, 1)] →
  (pf.map (λ p, p.2 + 1)).prod = 12 :=
by
  intros hn hpf
  rw [hn, hpf]
  simp
  sorry

end number_of_divisors_l384_384037


namespace complex_modulus_l384_384484

theorem complex_modulus (z : ℂ) (h : z * (2 + complex.i) = 5 * complex.i) : complex.abs (z - 1) = 2 :=
sorry

end complex_modulus_l384_384484


namespace volume_tetrahedrons_l384_384813

theorem volume_tetrahedrons (ABCD : Tetrahedron) (D1 : Point)
  (hD1ABC : is_centroid D1 ABC)
  (A1 B1 C1 : Point)
  (hA1 : inter_plane_opposite_vertex A DD1 == A1)
  (hB1 : inter_plane_opposite_vertex B DD1 == B1)
  (hC1 : inter_plane_opposite_vertex C DD1 == C1)
  : volume ABCD = (1 / 3) * volume (Tetrahedron A1 B1 C1 D1) :=
sorry

end volume_tetrahedrons_l384_384813


namespace no_integers_satisfying_polynomials_l384_384123

theorem no_integers_satisfying_polynomials 
: ¬ ∃ (a b c d : ℤ), a * 19^3 + b * 19^2 + c * 19 + d = 1 ∧ a * 62^3 + b * 62^2 + c * 62 + d = 2 := 
by
  sorry

end no_integers_satisfying_polynomials_l384_384123


namespace min_m_value_arithmetic_seq_l384_384069

theorem min_m_value_arithmetic_seq :
  ∀ (a S : ℕ → ℚ) (m : ℕ),
  (∀ n : ℕ, a (n+2) = 5 ∧ a (n+6) = 21) →
  (∀ n : ℕ, S (n+1) = S n + 1 / a (n+1)) →
  (∀ n : ℕ, S (2 * n + 1) - S n ≤ m / 15) →
  ∀ n : ℕ, m = 5 :=
sorry

end min_m_value_arithmetic_seq_l384_384069


namespace total_fencing_cost_is_correct_l384_384050

-- Define the fencing cost per side
def costPerSide : Nat := 69

-- Define the number of sides for a square
def sidesOfSquare : Nat := 4

-- Define the total cost calculation for fencing the square
def totalCostOfFencing (costPerSide : Nat) (sidesOfSquare : Nat) := costPerSide * sidesOfSquare

-- Prove that for a given cost per side and number of sides, the total cost of fencing the square is 276 dollars
theorem total_fencing_cost_is_correct : totalCostOfFencing 69 4 = 276 :=
by
    -- Proof goes here
    sorry

end total_fencing_cost_is_correct_l384_384050


namespace malou_average_score_l384_384488

-- Define the quiz scores as a list of real numbers.
def quizScores : List ℝ := [91, 90, 92, 87.5, 89.3, 94.7]

-- Define the function to calculate the average of a list of real numbers.
def average (scores : List ℝ) : ℝ := scores.sum / scores.length

-- State the theorem we want to prove.
theorem malou_average_score : average quizScores = 90.75 :=
by 
  -- Proof is omitted.
  sorry

end malou_average_score_l384_384488


namespace probability_no_two_counters_share_row_or_column_l384_384899

theorem probability_no_two_counters_share_row_or_column :
  let total_ways := Nat.choose 36 3,
      ways_no_overlap := 20 * 20 * 6,
      probability := ways_no_overlap.to_rat / total_ways.to_rat
  in probability = (40 : ℚ) / 119 :=
by
  let total_ways := Nat.choose 36 3
  let ways_no_overlap := 20 * 20 * 6
  let probability := ways_no_overlap.to_rat / total_ways.to_rat
  exact congr_arg (probability = _) sorry

end probability_no_two_counters_share_row_or_column_l384_384899


namespace find_b_for_continuity_l384_384103

def g (x : ℝ) (b : ℝ) : ℝ :=
  if x ≤ 3 then 3 * x^2 + 1 else b * x + 6

theorem find_b_for_continuity (b : ℝ) : 
  (∀ ε > 0, ∃ δ > 0, ∀ x, | x - 3 | < δ → | g x b - g 3 b | < ε) → 
  b = 22 / 3 :=
by
  sorry

end find_b_for_continuity_l384_384103


namespace divide_and_add_l384_384989

variable (number : ℝ)

theorem divide_and_add (h : 4 * number = 166.08) : number / 4 + 0.48 = 10.86 := by
  -- assume the proof follows accurately
  sorry

end divide_and_add_l384_384989


namespace area_of_triangle_ABC_eq_3_l384_384772

variable {n : ℕ}

def arithmetic_seq (a_1 d : ℤ) : ℕ → ℤ
| 0     => 0
| (n+1) => a_1 + n * d

def sum_arithmetic_seq (a_1 d : ℤ) : ℕ → ℤ
| 0     => 0
| (n+1) => (n + 1) * a_1 + (n * (n + 1) / 2) * d

def f (n : ℕ) : ℤ := sum_arithmetic_seq 4 6 n

def point_A (n : ℕ) : ℤ × ℤ := (n, f n)
def point_B (n : ℕ) : ℤ × ℤ := (n + 1, f (n + 1))
def point_C (n : ℕ) : ℤ × ℤ := (n + 2, f (n + 2))

def area_of_triangle (A B C : ℤ × ℤ) : ℤ :=
  (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)).natAbs / 2

theorem area_of_triangle_ABC_eq_3 : 
  ∀ (n : ℕ), area_of_triangle (point_A n) (point_B n) (point_C n) = 3 := 
sorry

end area_of_triangle_ABC_eq_3_l384_384772


namespace oysters_eaten_l384_384201

-- Define the conditions in Lean
def Squido_oysters : ℕ := 200
def Crabby_oysters (Squido_oysters : ℕ) : ℕ := 2 * Squido_oysters

-- Statement to prove
theorem oysters_eaten (Squido_oysters Crabby_oysters : ℕ) (h1 : Crabby_oysters = 2 * Squido_oysters) : 
  Squido_oysters + Crabby_oysters = 600 :=
by
  sorry

end oysters_eaten_l384_384201


namespace friedas_probability_l384_384898

theorem friedas_probability :
  let states := (Fin 4) × (Fin 4)
  let initial_state : states := (2 - 1, 1 - 1)
  let last_row : Set states := { ⟨3, 0⟩, ⟨3, 1⟩, ⟨3, 2⟩, ⟨3, 3⟩ }
  let move_possible (s : states) : Set states :=
    { (s.1, (s.2 + 1) % 4), ((s.1 + 1) % 4, s.2), (s.1, if s.2 = 0 then 3 else s.2 - 1) }
  let probability_of_stopping (hops : ℕ) : ℝ := sorry  -- Probability calculation
in probability_of_stopping 4 = 16 / 81 :=
sorry

end friedas_probability_l384_384898


namespace total_chewing_gums_l384_384517

-- Definitions for the conditions
def mary_gums : Nat := 5
def sam_gums : Nat := 10
def sue_gums : Nat := 15

-- Lean 4 Theorem statement to prove the total chewing gums
theorem total_chewing_gums : mary_gums + sam_gums + sue_gums = 30 := by
  sorry

end total_chewing_gums_l384_384517


namespace second_trial_amount_is_809_l384_384648

-- Define the lower and upper limits
def lower_limit : ℝ := 500
def upper_limit : ℝ := 1000

-- Define the amount added in the second trial using 0.618 method
def second_trial_amount : ℝ := lower_limit + (upper_limit - lower_limit) * 0.618

-- Prove that the amount added in the second trial is equal to 809 g
theorem second_trial_amount_is_809 : second_trial_amount = 809 :=
by
  -- calculate intermediate steps and intended final value
  have diff : ℝ := upper_limit - lower_limit
  have mult : ℝ := diff * 0.618
  have trial_amount : ℝ := lower_limit + mult
  -- the trial_amount should be 809
  show trial_amount = 809
  sorry

end second_trial_amount_is_809_l384_384648


namespace exponent_proof_l384_384351

theorem exponent_proof (n m : ℕ) (h1 : 4^n = 3) (h2 : 8^m = 5) : 2^(2*n + 3*m) = 15 :=
by
  -- Proof steps
  sorry

end exponent_proof_l384_384351


namespace part_a_part_b_l384_384208

-- Given the conditions:
-- 1. Two circles γ₁ and γ₂ intersect at points A and B.
-- 2. A line r passing through B intersects γ₁ at C and γ₂ at D, with B between C and D.
-- 3. Line s is parallel to AD and tangent to γ₁ at E, at the smaller distance from AD.
-- 4. Line EA intersects γ₂ at F.
-- 5. Line t is tangent to γ₂ at F.

variables {A B C D E F : Point} {γ₁ γ₂ : Circle}
variables (r s t : Line)

axiom circles_intersect : γ₁.intersects A B ∧ γ₂.intersects A B
axiom line_r_condition : r.passes_through B ∧ r.intersects γ₁ at C ∧ r.intersects γ₂ at D ∧ B is_between C D
axiom line_s_condition : s.parallel_to AD ∧ s.tangent_to γ₁ at E ∧ s.smaller_distance AD
axiom line_EA_intersect_F : Line_passing_through E A ∧ intersects γ₂ at F
axiom line_t_condition : t.tangent_to γ₂ at F

-- Part (a) Prove t is parallel to AC
theorem part_a : t.parallel_to (line_passing_through A C) :=
by
  sorry
-- Part (b) Prove the lines r, s, t are concurrent
theorem part_b : ∃ X, X ∈ r ∧ X ∈ s ∧ X ∈ t :=
by
  sorry

end part_a_part_b_l384_384208


namespace parabola_satisfies_given_condition_l384_384525

variable {p : ℝ}
variable {x1 x2 : ℝ}

-- Condition 1: The equation of the parabola is y^2 = 2px where p > 0.
def parabola_equation (p : ℝ) (x y : ℝ) : Prop :=
  y^2 = 2 * p * x

-- Condition 2: The parabola has a focus F.
-- Condition 3: A line passes through the focus F with an inclination angle of π/3.
def line_through_focus (p : ℝ) (x y : ℝ) : Prop :=
  y = Real.sqrt 3 * (x - p / 2)

-- Condition 4 & 5: The line intersects the parabola at points A and B with distance |AB| = 8.
def intersection_points (p : ℝ) (x1 x2 : ℝ) : Prop :=
  x1 ≠ x2 ∧ parabola_equation p x1 (Real.sqrt 3 * (x1 - p / 2)) ∧ parabola_equation p x2 (Real.sqrt 3 * (x2 - p / 2)) ∧
  abs (x1 - x2) * Real.sqrt (1 + 3) = 8

-- The proof statement
theorem parabola_satisfies_given_condition (hp : 0 < p) (hintersect : intersection_points p x1 x2) : 
  parabola_equation 3 x1 (Real.sqrt 3 * (x1 - 3 / 2)) ∧ parabola_equation 3 x2 (Real.sqrt 3 * (x2 - 3 / 2)) := sorry

end parabola_satisfies_given_condition_l384_384525


namespace smoking_related_to_lung_disease_l384_384076

theorem smoking_related_to_lung_disease (K2 : ℝ) :
  (K2 ≥ 6.635 → (99% confidence that smoking is related to lung disease)) ∧ 
  (K2 ≥ 6.635 → (1% chance of making a wrong judgment about the relationship between smoking and lung disease)) :=
by {
  sorry
}

end smoking_related_to_lung_disease_l384_384076


namespace total_oysters_eaten_l384_384199

/-- Squido eats 200 oysters -/
def Squido_eats := 200

/-- Crabby eats at least twice as many oysters as Squido -/
def Crabby_eats := 2 * Squido_eats

/-- Total oysters eaten by Squido and Crabby -/
theorem total_oysters_eaten : Squido_eats + Crabby_eats = 600 := 
by
  sorry

end total_oysters_eaten_l384_384199


namespace find_angle_AXC_l384_384848

theorem find_angle_AXC (ABCD : Quadrilateral) (X A D B C E : Point)
  (h1 : Convex ABCD)
  (h2 : OnSegment X A D)
  (h3 : Intersect AC BD E)
  (h4 : Length AC = Length BD)
  (h5 : angle A B X = 50 ∧ angle A X B = 50)
  (h6 : angle C A D = 51)
  (h7 : angle A E D = 80)
  : angle A X C = 80 :=
sorry

end find_angle_AXC_l384_384848


namespace num_terminating_decimals_l384_384679

theorem num_terminating_decimals (n : ℕ) (h1 : 1 ≤ n ∧ n ≤ 180) :
  (∃ k : ℕ, k = 20) := by
  have h2 : ∀ n, (1 ≤ n ∧ n ≤ 180) → (∃ k, k = (180 / 9)) := sorry
  have h3 : 180 / 9 = 20 := by norm_num
  use 20
  exact h3

end num_terminating_decimals_l384_384679


namespace power_sum_is_integer_l384_384121

theorem power_sum_is_integer (a : ℝ) (n : ℕ) (h_pos : 0 < n)
  (h_k : ∃ k : ℤ, k = a + 1/a) : 
  ∃ m : ℤ, m = a^n + 1/a^n := 
sorry

end power_sum_is_integer_l384_384121


namespace terminating_decimal_count_l384_384745

theorem terminating_decimal_count : 
  (Finset.filter (λ n, n % 3 = 0) (Finset.range 181)).card = 60 := by
  sorry

end terminating_decimal_count_l384_384745


namespace divisors_of_90_l384_384034

def num_pos_divisors (n : ℕ) : ℕ :=
  let factors := if n = 90 then [(2, 1), (3, 2), (5, 1)] else []
  factors.foldl (fun acc (p, k) => acc * (k + 1)) 1

theorem divisors_of_90 : num_pos_divisors 90 = 12 := by
  sorry

end divisors_of_90_l384_384034


namespace sandy_initial_payment_l384_384504

theorem sandy_initial_payment (P : ℝ) (H1 : P + 300 < P + 1320)
  (H2 : 1320 = 1.10 * (P + 300)) : P = 900 :=
sorry

end sandy_initial_payment_l384_384504


namespace smallest_possible_sum_l384_384371

theorem smallest_possible_sum (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (hneq : a ≠ b) 
  (heq : (1 / a : ℚ) + (1 / b) = 1 / 12) : a + b = 49 :=
sorry

end smallest_possible_sum_l384_384371


namespace largest_angle_right_triangle_l384_384174

theorem largest_angle_right_triangle (u : ℝ) (h1 : 3u - 2 > 0) (h2 : 3u + 2 > 0) : 
  ∃ (A B C : ℝ), (A = sqrt (3u - 2)) ∧ (B = sqrt (3u + 2)) ∧ (C = 2 * sqrt u) ∧ 
  (A^2 + B^2 = C^2) ∧ (angle C = 90) :=
by
-- Definitions based on conditions
have sq_A : A^2 = 3u - 2 := by sorry
have sq_B : B^2 = 3u + 2 := by sorry
have sq_C : C^2 = 4u := by sorry

-- Proofs of inequalities and equalities based on side lengths
-- Triangle inequality conditions
have tri_ineq_1 : A + B > C := by sorry
have tri_ineq_2 : A + C > B := by sorry
have tri_ineq_3 : B + C > A := by sorry

-- Pythagorean theorem condition
have right_triangle : 3u - 2 + 3u + 2 = 4u + 2u := by sorry

-- Checking largest angle
have angle_largest : angle C = 90 := by sorry

-- Conclusion
existsi (sqrt (3u - 2)), existsi (sqrt (3u + 2)), existsi (2 * sqrt u),
simp [sq_A, sq_B, sq_C, angle C, right_triangle],
sorry

end largest_angle_right_triangle_l384_384174


namespace area_of_T_eq_3sqrt3_l384_384479

noncomputable def omega : ℂ := -1/2 + (1/2) * complex.I * real.sqrt 3

def T : set ℂ :=
  {z | ∃ (a b c : ℝ), 0 ≤ a ∧ a ≤ 1 ∧ 0 ≤ b ∧ b ≤ 1 ∧ 0 ≤ c ∧ c ≤ 1 ∧
    z = 2 * a + b * omega + c * omega^2}

theorem area_of_T_eq_3sqrt3 : measure_theory.measure.space volume (T) = 3 * real.sqrt 3 :=
sorry

end area_of_T_eq_3sqrt3_l384_384479


namespace inequality_III_l384_384990

variable (a b c x y z : ℝ)
variable (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
variable (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
variable (hxa : x ≤ a) (hyb : y ≤ b) (hzc : z ≤ c)

theorem inequality_III (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
                       (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
                       (hxa : x ≤ a) (hyb : y ≤ b) (hzc : z ≤ c) :
  sqrt x + sqrt y + sqrt z < sqrt a + sqrt b + sqrt c :=
sorry

end inequality_III_l384_384990


namespace scientific_notation_of_0_00000002_l384_384515

theorem scientific_notation_of_0_00000002 : (0.00000002 : ℝ) = 2 * 10^(-8) :=
by
  sorry

end scientific_notation_of_0_00000002_l384_384515


namespace polynomial_coeff_sum_l384_384462

theorem polynomial_coeff_sum :
  let a : ℕ → ℤ := λ n, polynomial.coeff (polynomial.C (2 : ℤ) - polynomial.C (1 : ℤ) * polynomial.X)^6 n in
  |a 1| + |a 2| + |a 3| + |a 4| + |a 5| + |a 6| = 665 :=
by
  let a : ℕ → ℤ := λ n, polynomial.coeff (polynomial.C (2 : ℤ) - polynomial.C (1 : ℤ) * polynomial.X)^6 n
  sorry

end polynomial_coeff_sum_l384_384462


namespace cone_circumference_l384_384606

theorem cone_circumference (V h : ℝ) (V_eq : V = 24 * π) (h_eq : h = 6) :
  ∃ C : ℝ, C = 4 * real.sqrt 3 * π :=
by {
  let r := real.sqrt 12,
  have r_eq : r = 2 * real.sqrt 3 := sorry,
  let C := 2 * π * r,
  use C,
  rw [r_eq],
  linarith,
  sorry
}

#exit

end cone_circumference_l384_384606


namespace jeffery_fish_count_l384_384862

variable (J R Y : ℕ)

theorem jeffery_fish_count :
  (R = 3 * J) → (Y = 2 * R) → (J + R + Y = 100) → (Y = 60) :=
by
  intros hR hY hTotal
  have h1 : R = 3 * J := hR
  have h2 : Y = 2 * R := hY
  rw [h1, h2] at hTotal
  sorry

end jeffery_fish_count_l384_384862


namespace student_average_grade_l384_384575

theorem student_average_grade
  (courses_year1 : ℕ)
  (avg_grade_year1 : ℝ)
  (courses_year2 : ℕ)
  (avg_grade_year2 : ℝ)
  (h1 : courses_year1 = 6)
  (h2 : avg_grade_year1 = 100)
  (h3 : courses_year2 = 5)
  (h4 : avg_grade_year2 = 70) :
  let total_points := (courses_year1 * avg_grade_year1) + (courses_year2 * avg_grade_year2)
  let total_courses := courses_year1 + courses_year2
  let avg_grade_period := total_points / total_courses
  avg_grade_period ≈ 86.4 :=
by
  sorry

end student_average_grade_l384_384575


namespace ramu_original_cost_l384_384125

noncomputable def original_cost (C : ℝ) : Prop :=
  let total_cost := C + 13000
  let selling_price := 61900
  let profit_percent := 12.545454545454545 / 100
  let profit := selling_price - total_cost
  profit = C * profit_percent

theorem ramu_original_cost : original_cost 43455 :=
by
  let C := 43455
  let total_cost := C + 13000
  let selling_price := 61900
  let profit_percent := 12.545454545454545 / 100
  let profit := selling_price - total_cost
  have h : profit = C * profit_percent := sorry
  exact h

end ramu_original_cost_l384_384125


namespace smallest_solution_proof_l384_384336

noncomputable def smallest_solution : ℝ :=
  let n := 11
  let a := 0.533
  n + a

theorem smallest_solution_proof :
  ∃ (x : ℝ), ⌊x^2⌋ - ⌊x⌋^2 = 21 ∧ x = smallest_solution :=
by
  use smallest_solution
  sorry

end smallest_solution_proof_l384_384336


namespace equilateral_triangle_condition_l384_384154

-- We define points in a plane and vectors between these points
structure Point where
  x : ℝ
  y : ℝ

-- Vector subtraction
def vector (p q : Point) : Point :=
  { x := q.x - p.x, y := q.y - p.y }

-- The equation required to hold for certain type of triangles
def bisector_eq_zero (A B C A1 B1 C1 : Point) : Prop :=
  let AA1 := vector A A1
  let BB1 := vector B B1
  let CC1 := vector C C1
  AA1.x + BB1.x + CC1.x = 0 ∧ AA1.y + BB1.y + CC1.y = 0

-- Property of equilateral triangle
def is_equilateral (A B C : Point) : Prop :=
  let AB := vector A B
  let BC := vector B C
  let CA := vector C A
  (AB.x^2 + AB.y^2 = BC.x^2 + BC.y^2 ∧ BC.x^2 + BC.y^2 = CA.x^2 + CA.y^2)

-- Main theorem statement
theorem equilateral_triangle_condition (A B C A1 B1 C1 : Point)
  (h : bisector_eq_zero A B C A1 B1 C1) :
  is_equilateral A B C :=
sorry

end equilateral_triangle_condition_l384_384154


namespace exists_d_for_m_divides_f_of_f_n_l384_384460

noncomputable def f : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => 23 * f (n + 1) + f n

theorem exists_d_for_m_divides_f_of_f_n (m : ℕ) : 
  ∃ (d : ℕ), ∀ (n : ℕ), m ∣ f (f n) ↔ d ∣ n := 
sorry

end exists_d_for_m_divides_f_of_f_n_l384_384460


namespace factor_t_squared_minus_81_l384_384318

theorem factor_t_squared_minus_81 (t : ℂ) : (t^2 - 81) = (t - 9) * (t + 9) := 
by
  -- We apply the identity a^2 - b^2 = (a - b) * (a + b)
  let a := t
  let b := 9
  have eq : t^2 - 81 = a^2 - b^2 := by sorry
  rw [eq]
  exact (mul_sub_mul_add_eq_sq_sub_sq a b).symm
  -- Concluding the proof
  sorry -- skipping detailed proof steps for now

end factor_t_squared_minus_81_l384_384318


namespace inverse_function_passes_through_fixed_point_inv_l384_384249

noncomputable theory

variables {a : ℝ} (ha_pos : 0 < a) (ha_ne_one : a ≠ 1)

def f (x : ℝ) : ℝ := log a (x + 3)

def inv_f (y : ℝ) : ℝ := a ^ y - 3

theorem inverse_function_passes_through_fixed_point_inv (h_pos : 0 < a) (h_ne : a ≠ 1) :
  inv_f a (f 0) = -2 :=
by
  have h1 : ∀ y, inv_f a y = a ^ y - 3, from sorry
  have h2 : inv_f a 0 = -3, from sorry
  have h3 : f 0 = log a 3, from sorry
  rw [h1, h3]
  sorry

end inverse_function_passes_through_fixed_point_inv_l384_384249


namespace solution_set_inequality_l384_384013

-- Definitions for function and properties
variable {R : Type*} [TopologicalSpace R] [Nonempty R] [Preorder R] [OrderClosedTopology R]
variable (f : R → R)
variable dom_f : R → Prop
variable (f_deriv : R → R) (h_domain : ∀ x, dom_f x ↔ x > 0)
variable (h_deriv : ∀ x, x * f_deriv x - 1 < 0)
variable (h_fe : f (exp 1) = 2)

-- Theorem statement for proof
theorem solution_set_inequality :
  { x : R | f (exp x) < x + 1 } = { x : R | x > 1 } := by sorry

end solution_set_inequality_l384_384013


namespace proof_value_of_x_l384_384445

-- Definitions for the initial conditions
variables {P Q R S T : Type} [Points P Q R S T]

-- Angle measure definitions
def angle_PQS := 150 -- degrees
def angle_PTS := 72 -- degrees

-- Theorem: Given the conditions in a), prove that x = 78 degrees.
theorem proof_value_of_x (PQRS_is_quadrilateral : Quadrilateral PQRS)
  (PQR_straight_line : StraightLine P Q R)
  (angle_PQS_eq: angle_PQS = 150)
  (angle_PTS_eq: angle_PTS = 72)
  (angle_PQS_exterior_of_ΔQTS : ExteriorAngle_of_Triangle PQS QTS) :
  x = 78 :=
by
  sorry -- Proof is omitted

end proof_value_of_x_l384_384445


namespace students_in_lucas_graduating_class_l384_384523

theorem students_in_lucas_graduating_class :
  ∃ (x : ℕ), 70 < x ∧ x < 120 ∧
  (∃ k : ℤ, x = 6 * k - 2) ∧
  (∃ m : ℤ, x = 5 * m - 3) ∧
  (∃ n : ℤ, x = 7 * n - 4) ∧
  x = 148 :=
begin
  sorry
end

end students_in_lucas_graduating_class_l384_384523


namespace solve_problem_l384_384015

namespace Example

-- Definitions based on given conditions
def isEvenFunction (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

def condition_2 (f : ℝ → ℝ) : Prop := f 2 = -1

def condition_3 (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = -f (2 - x)

-- Main theorem statement
theorem solve_problem (f : ℝ → ℝ)
  (h1 : isEvenFunction f)
  (h2 : condition_2 f)
  (h3 : condition_3 f) : f 2016 = 1 :=
sorry

end Example

end solve_problem_l384_384015


namespace minimal_disks_needed_l384_384107

theorem minimal_disks_needed (num_files : ℕ) 
  (file1_size : ℕ) (num_file1 : ℕ)
  (file2_size : ℕ) (num_file2 : ℕ)
  (file3_size : ℕ) (disk_capacity : ℕ)
  (remaining_files_count : ℕ) :
  num_files = 36 →
  file1_size = 12 →
  num_file1 = 4 →
  file2_size = 6 →
  num_file2 = 16 →
  file3_size = 5 →
  disk_capacity = 20 →
  remaining_files_count = num_files - num_file1 - num_file2 →
  ∃ (total_disks : ℕ), total_disks = 12 :=
by
  intro h1 h2 h3 h4 h5 h6 h7 h8
  use 12
  sorry

end minimal_disks_needed_l384_384107


namespace solve_for_x_l384_384141

-- Define the problem
def equation (x : ℝ) : Prop := x + 2 * x + 12 = 500 - (3 * x + 4 * x)

-- State the theorem that we want to prove
theorem solve_for_x : ∃ (x : ℝ), equation x ∧ x = 48.8 := by
  sorry

end solve_for_x_l384_384141


namespace largest_value_of_c_l384_384101

theorem largest_value_of_c : ∀ c : ℝ, (3 * c + 6) * (c - 2) = 9 * c → c ≤ 4 :=
by
  intros c hc
  have : (3 * c + 6) * (c - 2) = 9 * c := hc
  sorry

end largest_value_of_c_l384_384101


namespace quadratic_two_distinct_real_roots_l384_384182

theorem quadratic_two_distinct_real_roots : 
  ∀ x : ℝ, ∃ a b c : ℝ, (∀ x : ℝ, (x+1)*(x-1) = 2*x + 3 → x^2 - 2*x - 4 = 0) ∧ 
  (a = 1) ∧ (b = -2) ∧ (c = -4) ∧ (b^2 - 4*a*c > 0) :=
by
  sorry

end quadratic_two_distinct_real_roots_l384_384182


namespace median_commutes_l384_384493

noncomputable def morning_commute_speeds : List ℕ := [24, 60, 62, 63, 65, 81, 86, 87, 89]
noncomputable def evening_commute_speeds : List ℕ := [60, 60, 62, 66, 67, 70, 71, 72, 72, 75, 77, 78]

theorem median_commutes :
  median morning_commute_speeds = 65 ∧ median evening_commute_speeds = 70.5 :=
by
  sorry

end median_commutes_l384_384493


namespace curve_C1_eqn_and_curve_C2_eqn_and_triangle_area_max_l384_384066

theorem curve_C1_eqn_and_curve_C2_eqn_and_triangle_area_max (
    α θ θ1 θ2 : ℝ
    (P Q : ℝ × ℝ)
    (hC1x : ∀ α, 2 + 2 * real.cos α = P.1)
    (hC1y : ∀ α, 2 * real.sin α = P.2)
    (hC2 : ∀ θ, 2 * real.cos θ = Q.1)
    (angle_POQ : θ1 = θ2 + (π / 2))
    (hP : P.1 = 2 + 2 * real.cos θ1 ∧ P.2 = 2 * real.sin θ1)
    (hQ : Q.1 = 2 * real.cos θ2)
    (hρ1 : ∀ θ1, P.1 = 4 * real.cos θ1)
    (hρ2 : ∀ θ2, Q.1 = 2 * real.cos θ2)) :
  ∃ x y : ℝ, 
    ((x - 2)^2 + y^2 = 4) ∧ 
    ((x - 1)^2 + y^2 = 1) ∧ 
    (∃ (θ2 : ℝ), -2 * real.sin(2 * θ2) ≤ 2) :=
sorry

end curve_C1_eqn_and_curve_C2_eqn_and_triangle_area_max_l384_384066


namespace sample_size_stratified_sampling_l384_384607

theorem sample_size_stratified_sampling
  (students : ℕ) (freshmen : ℕ) (sample_freshmen : ℕ) (n : ℕ)
  (h1 : students = 1500) (h2 : freshmen = 400) (h3 : sample_freshmen = 12)
  (h4 : (sample_freshmen : ℚ) / freshmen = (n : ℚ) / students) :
  n = 45 :=
by
  -- Conditions are translated faithfully into the theorem
  rw [h1, h2, h3] at h4,
  -- Perform necessary proof steps (which we omit as the prompt asks)
  sorry

end sample_size_stratified_sampling_l384_384607


namespace probability_A_correct_l384_384049

-- Define the events
variables (A B : Prop)

-- Define the probabilities as assumptions
variable (P_B : ℝ)
variable (P_A_and_B : ℝ)
variable (P_neither : ℝ)

-- Given conditions
axiom pb_def : P_B = 0.55
axiom pab_def : P_A_and_B = 0.45
axiom pneither_def : P_neither = 0.20

-- Define the probability of A
noncomputable def P_A : ℝ :=
  0.8 - P_B + P_A_and_B

-- Theorem to prove
theorem probability_A_correct : P_A = 0.70 :=
by
  rw [pb_def, pab_def, pneither_def]
  sorry

end probability_A_correct_l384_384049


namespace proof_problem_l384_384625

-- List of body weight increases in control group
def controlGroup : List ℝ := [15.2, 18.8, 20.2, 21.3, 22.5, 23.2, 25.8, 26.5, 27.5, 30.1,
                              32.6, 34.3, 34.8, 35.6, 35.6, 35.8, 36.2, 37.3, 40.5, 43.2]

-- List of body weight increases in experimental group
def experimentalGroup : List ℝ := [7.8, 9.2, 11.4, 12.4, 13.2, 15.5, 16.5, 18.0, 18.8, 19.2,
                                   19.8, 20.2, 21.6, 22.8, 23.6, 23.9, 25.1, 28.2, 32.3, 36.5]

-- Sample mean of the experimental group
noncomputable def sampleMeanExperimental : ℝ := (experimentalGroup.sum) / (experimentalGroup.length)

-- Median m of the combined list
def combinedGroup : List ℝ := (controlGroup ++ experimentalGroup).sort
noncomputable def median : ℝ := (combinedGroup.get! 19 + combinedGroup.get! 20) / 2

-- Contingency table counts
def controlLessThanM : ℕ := controlGroup.filter (λ x => x < median).length
def controlGreaterThanOrEqualM : ℕ := controlGroup.filter (λ x => x >= median).length
def experimentalLessThanM : ℕ := experimentalGroup.filter (λ x => x < median).length
def experimentalGreaterThanOrEqualM : ℕ := experimentalGroup.filter (λ x => x >= median).length

-- Chi-squared value K^2
noncomputable def K2 : ℝ :=
  let a := controlLessThanM
  let b := controlGreaterThanOrEqualM
  let c := experimentalLessThanM
  let d := experimentalGreaterThanOrEqualM
  let n := controlGroup.length + experimentalGroup.length
  n * (a * d - b * c) ^ 2 / ((a + b) * (c + d) * (a + c) * (b + d))

theorem proof_problem :
  sampleMeanExperimental = 19.8 ∧
  median = 23.4 ∧
  controlLessThanM = 6 ∧ controlGreaterThanOrEqualM = 14 ∧
  experimentalLessThanM = 14 ∧ experimentalGreaterThanOrEqualM = 6 ∧
  K2 = 6.4 :=
by
  unfold sampleMeanExperimental median controlLessThanM controlGreaterThanOrEqualM
         experimentalLessThanM experimentalGreaterThanOrEqualM K2
  simp only [*, List.length, List.sum, List.filter, List.sort, List.get!]
  -- Add necessary calculations to verify each part
  sorry

end proof_problem_l384_384625


namespace compute_xy_l384_384978

theorem compute_xy (x y : ℝ) (h1 : x - y = 6) (h2 : x^2 - y^2 = 20) : x * y = -56 / 9 :=
by
  sorry

end compute_xy_l384_384978


namespace solve_inequality_l384_384016

noncomputable def f : ℝ → ℝ := sorry  -- Definition of f is not provided, so we use sorry

axiom f_mono_increasing : ∀ x y : ℝ, x < y → f x < f y 
axiom f_functional_eq : ∀ x : ℝ, f(5 - x) = -f(-1 + x)

theorem solve_inequality : ∀ x : ℝ, (| f (x^2 - 2x + 7) | < f (x^2 + 3x + 2)) → x > 1 :=
sorry

end solve_inequality_l384_384016


namespace terminating_decimal_count_l384_384749

theorem terminating_decimal_count : 
  (Finset.filter (λ n, n % 3 = 0) (Finset.range 181)).card = 60 := by
  sorry

end terminating_decimal_count_l384_384749


namespace right_angle_triangle_l384_384872

noncomputable def a (a b c: ℂ) (k1: ℝ) : ℂ := a + k1 * I * (c - b)
noncomputable def b' (b a c: ℂ) (k2: ℝ) : ℂ := b + k2 * I * (a - c)
noncomputable def c' (c b a: ℂ) (k3: ℝ) : ℂ := c + k3 * I * (b - a)

theorem right_angle_triangle
  (ABC : Triangle)
  (k1 k2 k3 : ℝ)
  (A' B' C' : Point)
  (hA' : A' = a(ABC.A, ABC.B, ABC.C, k1))
  (hB' : B' = b'(ABC.B, ABC.A, ABC.C, k2))
  (hC' : C' = c'(ABC.C, ABC.B, ABC.A, k3))
  : ((B' - A') * (C' - A') = 0) ↔ (A'B'C' is right-angled at A') :=
sorry

end right_angle_triangle_l384_384872


namespace gain_per_year_l384_384273

-- Definitions of the conditions
def principal : ℝ := 5000
def rate1 : ℝ := 4
def rate2 : ℝ := 5
def time : ℝ := 2

-- Definition of simple interest calculation
def simple_interest (P R T : ℝ) := (P * R * T) / 100

-- Conditions encoded
def interest_paid := simple_interest principal rate1 time
def interest_earned := simple_interest principal rate2 time

-- Statement of the theorem
theorem gain_per_year :
  let gain := interest_earned - interest_paid
  gain / time = 50 :=
by
  sorry -- Proof is skipped

end gain_per_year_l384_384273


namespace least_difference_squared_l384_384190

theorem least_difference_squared (a1 a2 a3 a4 a5 : ℝ) 
  (h : a1^2 + a2^2 + a3^2 + a4^2 + a5^2 = 1) :
  ∃ i j, i ≠ j ∧ (a_i - a_j)^2 ≤ 1 / 10 :=
sorry

end least_difference_squared_l384_384190


namespace additional_boys_went_down_slide_l384_384976

theorem additional_boys_went_down_slide (initial_boys total_boys additional_boys : ℕ) (h1 : initial_boys = 22) (h2 : total_boys = 35) : additional_boys = 13 :=
by {
    -- Proof body will be here
    sorry
}

end additional_boys_went_down_slide_l384_384976


namespace area_of_triangle_G1G2G3_l384_384466

theorem area_of_triangle_G1G2G3 (P : Point) (A B C G1 G2 G3 : Point)
  (h_incenter : incenter P A B C)
  (h_G1 : centroid G1 P B C)
  (h_G2 : centroid G2 P C A)
  (h_G3 : centroid G3 P A B)
  (h_area : area A B C = 36) :
  area G1 G2 G3 = 4 :=
sorry

end area_of_triangle_G1G2G3_l384_384466


namespace problem_a_problem_b_l384_384756

-- Definition for real roots condition in problem A
def has_real_roots (k : ℝ) : Prop :=
  let a := 1
  let b := -3
  let c := k
  b^2 - 4 * a * c ≥ 0

-- Problem A: Proving the range of k
theorem problem_a (k : ℝ) : has_real_roots k ↔ k ≤ 9 / 4 :=
by
  sorry

-- Definition for a quadratic equation having a given root
def has_root (a b c : ℝ) (x : ℝ) : Prop :=
  a * x^2 + b * x + c = 0

-- Problem B: Proving the value of m given a common root condition
theorem problem_b (m : ℝ) : 
  (has_root 1 (-3) 2 1 ∧ has_root (m-1) 1 (m-3) 1) ↔ m = 3 / 2 :=
by
  sorry

end problem_a_problem_b_l384_384756


namespace num_terminating_decimals_l384_384680

theorem num_terminating_decimals (n : ℕ) (h1 : 1 ≤ n ∧ n ≤ 180) :
  (∃ k : ℕ, k = 20) := by
  have h2 : ∀ n, (1 ≤ n ∧ n ≤ 180) → (∃ k, k = (180 / 9)) := sorry
  have h3 : 180 / 9 = 20 := by norm_num
  use 20
  exact h3

end num_terminating_decimals_l384_384680


namespace largest_number_exists_l384_384991

theorem largest_number_exists :
  let A := 0.998
  let B := 0.989
  let C := 0.999
  let D := 0.990
  let E := 0.980
  C > A ∧ C > B ∧ C > D ∧ C > E :=
by
  -- Definitions based on the conditions
  let A := 0.998
  let B := 0.989
  let C := 0.999
  let D := 0.990
  let E := 0.980

  -- Proof starts here.
  have h1 : C > A := by norm_num
  have h2 : C > B := by norm_num
  have h3 : C > D := by norm_num
  have h4 : C > E := by norm_num
  exact ⟨h1, h2, h3, h4⟩

end largest_number_exists_l384_384991


namespace eliminate_denominators_l384_384615

theorem eliminate_denominators (x : ℚ) :
  (x + 1) / 2 + 1 = x / 3 → 3 * (x + 1) + 6 = 2 * x :=
begin
  intro h,
  sorry
end

end eliminate_denominators_l384_384615


namespace alpha_eq_beta_l384_384352

variable {α β : ℝ}

theorem alpha_eq_beta
  (h_alpha : 0 < α ∧ α < (π / 2))
  (h_beta : 0 < β ∧ β < (π / 2))
  (h_sin : Real.sin (α + β) + Real.sin (α - β) = Real.sin (2 * β)) :
  α = β :=
by
  sorry

end alpha_eq_beta_l384_384352


namespace range_of_a_l384_384811

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | 2 - a ≤ x ∧ x ≤ 2 + a}
def B : Set ℝ := {x | 0 ≤ x ∧ x ≤ 5}

-- Define the theorem to be proved
theorem range_of_a (a : ℝ) (h₁ : A a ⊆ B) (h₂ : 2 - a < 2 + a) : 0 < a ∧ a ≤ 2 :=
by
  sorry

end range_of_a_l384_384811


namespace spy_diagram_is_fake_l384_384281

def cityA_visits : ℕ := 50
def cityB_visits : ℕ := 30
def cityC_visits : ℕ := 20
def cityD_visits : ℕ := 70
def cityE_visits : ℕ := 60
def cityF_visits : ℕ := 40

def triangular_visits : ℕ := cityA_visits + cityB_visits + cityC_visits
def arch_visits : ℕ := cityD_visits + cityE_visits + cityF_visits

theorem spy_diagram_is_fake : triangular_visits < arch_visits - 1 :=
by
  have h1 : triangular_visits = 100 := by sorry
  have h2 : arch_visits = 170 := by sorry
  have h3 : arch_visits - 1 = 169 := by sorry
  show triangular_visits < arch_visits - 1 from sorry

end spy_diagram_is_fake_l384_384281


namespace cool_double_l384_384272

def is_cool (k : ℕ) : Prop :=
  ∃ (a b : ℕ), k = a^2 + b^2

theorem cool_double (k : ℕ) (h : is_cool k) : is_cool (2 * k) := by
  cases h with
  | intro a b hk =>
    use a - b, a + b
    rw [← hk, mul_add, add_mul, add_mul]
    dsimp
    rw [add_assoc, add_assoc, ← sq, ← sq, ← sq, ← sq, add_zero]
    apply congr_arg2
    · ring
    · ring
    done
  done
  sorry

end cool_double_l384_384272


namespace polynomial_even_iff_exists_Q_l384_384500

theorem polynomial_even_iff_exists_Q (P Q : Polynomial ℂ) :
  (∀ z : ℂ, P(z) = P(-z)) ↔ (∃ Q : Polynomial ℂ, ∀ z : ℂ, P(z) = Q(z) * Q(-z)) :=
sorry

end polynomial_even_iff_exists_Q_l384_384500


namespace mid_points_distance_relation_l384_384098

variable {A B C D E M : Type}
variables (triangle: ∀ (A B C : Type), Prop) [IsAcute ∈ triangle A B C]
variables (midpoint : ∀ (P Q : Type), Type)
variables [mid_AB : midpoint A B D] [mid_BC : midpoint B C E]
variables (onSegment : ∀ (P Q R : Type), Prop)
variables [on_AC : onSegment A C M]
variables (dist : ∀ (P Q : Type), Type)
variable [dist_MD : dist M D] [dist_AD : dist A D] [dist_ME : dist M E] [dist_EC : dist E C]

theorem mid_points_distance_relation (h : dist_MD < dist_AD) : dist_ME > dist_EC := 
sorry

end mid_points_distance_relation_l384_384098


namespace evaluate_fg_of_2_l384_384047

def f (x : ℝ) : ℝ := x ^ 3
def g (x : ℝ) : ℝ := 4 * x + 5

theorem evaluate_fg_of_2 : f (g 2) = 2197 :=
by
  sorry

end evaluate_fg_of_2_l384_384047


namespace terminating_decimal_count_l384_384751

theorem terminating_decimal_count : 
  (Finset.filter (λ n, n % 3 = 0) (Finset.range 181)).card = 60 := by
  sorry

end terminating_decimal_count_l384_384751


namespace valid_choice_count_l384_384631

def is_valid_base_7_digit (n : ℕ) : Prop := n < 7
def is_valid_base_8_digit (n : ℕ) : Prop := n < 8
def to_base_10_base_7 (c3 c2 c1 c0 : ℕ) : ℕ := 2401 * c3 + 343 * c2 + 49 * c1 + 7 * c0
def to_base_10_base_8 (d3 d2 d1 d0 : ℕ) : ℕ := 4096 * d3 + 512 * d2 + 64 * d1 + 8 * d0
def is_four_digit_number (n : ℕ) : Prop := n >= 1000 ∧ n < 10000

theorem valid_choice_count :
  ∃ (N : ℕ), is_four_digit_number N →
  ∀ (c3 c2 c1 c0 d3 d2 d1 d0 : ℕ),
    is_valid_base_7_digit c3 → is_valid_base_7_digit c2 → is_valid_base_7_digit c1 → is_valid_base_7_digit c0 →
    is_valid_base_8_digit d3 → is_valid_base_8_digit d2 → is_valid_base_8_digit d1 → is_valid_base_8_digit d0 →
    to_base_10_base_7 c3 c2 c1 c0 = N →
    to_base_10_base_8 d3 d2 d1 d0 = N →
    (to_base_10_base_7 c3 c2 c1 c0 + to_base_10_base_8 d3 d2 d1 d0) % 1000 = (2 * N) % 1000 → N = 20 :=
sorry

end valid_choice_count_l384_384631


namespace expected_number_of_edges_same_color_3x3_l384_384051

noncomputable def expected_edges_same_color (board_size : ℕ) (blackened_count : ℕ) : ℚ :=
  let total_pairs := 12       -- 6 horizontal pairs + 6 vertical pairs
  let prob_both_white := 1 / 6
  let prob_both_black := 5 / 18
  let prob_same_color := prob_both_white + prob_both_black
  total_pairs * prob_same_color

theorem expected_number_of_edges_same_color_3x3 :
  expected_edges_same_color 3 5 = 16 / 3 :=
by
  sorry

end expected_number_of_edges_same_color_3x3_l384_384051


namespace available_codes_for_reckha_l384_384490

-- Constants and assumptions
def code := ℕ × ℕ × ℕ

def digits_valid (d : ℕ) := d ≥ 0 ∧ d < 10

def is_valid_digit (d : ℕ) : Prop := digits_valid d

def has_even_digit (c : code) : Prop :=
  let (a, b, c) := c in (a % 2 = 0 ∨ b % 2 = 0 ∨ c % 2 = 0)

def does_not_match_145 (c : code) : Prop :=
  let (a, b, c) := c in
  (a ≠ 1 ∨ b ≠ 4) ∧ (a ≠ 1 ∨ c ≠ 5) ∧ (b ≠ 4 ∨ c ≠ 5)

def does_not_match_exactly_145 (c : code) : Prop :=
  let (a, b, c) := c in ¬(a = 1 ∧ b = 4 ∧ c = 5)

-- Theorem to prove
theorem available_codes_for_reckha : ∃ n : ℕ, n = 844 ∧
  (n = (Finset.card (
    (Finset.product
      (Finset.filter is_valid_digit (Finset.range 10))
      (Finset.product (Finset.filter is_valid_digit (Finset.range 10)) (Finset.filter is_valid_digit (Finset.range 10)))))
  ).card - (Finset.filter does_not_match_145 (Finset.fromList {(1, 4, 5)})).card - (Finset.singleton (1, 4, 5)).card)) :=
by
  -- the proof is omitted but should establish the given condition and computed codes.
  sorry

end available_codes_for_reckha_l384_384490


namespace probability_within_radius_l384_384604

structure Rectangle where
  x1 y1 x2 y2 : ℝ
  h_pos : 0 ≤ x1 ∧ 0 ≤ y1 ∧ x1 ≤ x2 ∧ y1 ≤ y2

def is_lattice_point (p : ℝ × ℝ) : Prop :=
  ∃ (m n : ℤ), p = (m, n)

noncomputable def lattice_point_probability (r : Rectangle) (d : ℝ) : ℝ :=
  let unit_square_area := 1
  let lattice_influence_area := π * d ^ 2
  lattice_influence_area / unit_square_area

theorem probability_within_radius (r : Rectangle) (d : ℝ) :
  r = { x1 := 0, y1 := 0, x2 := 3030, y2 := 2020, h_pos := sorry } →
  lattice_point_probability r d = 1 / 3 →
  d = 0.3 :=
sorry

end probability_within_radius_l384_384604


namespace smallest_consecutive_sum_l384_384188

theorem smallest_consecutive_sum (n : ℤ) (h : n + (n + 1) + (n + 2) + (n + 3) + (n + 4) = 210) : 
  n = 40 := 
sorry

end smallest_consecutive_sum_l384_384188


namespace evaluate_series_l384_384664

def closest_integer (x : ℝ) : ℤ :=
if h : (x - floor x < 0.5) then ⌊x⌋ else ⌊x⌋ + 1

def n_proximity (n : ℕ) := closest_integer (real.rpow (n ^ 3 : ℝ) (1/6))

theorem evaluate_series :
  ∑' (n : ℕ), (3 ^ (n_proximity n) + 3 ^ (- (n_proximity n))) / 3 ^ n = 1 := by
sorry

end evaluate_series_l384_384664


namespace cheese_cost_l384_384503

theorem cheese_cost
  (hamburger_meat : ℝ := 5.00)
  (crackers : ℝ := 3.50)
  (vegetables_unit_price : ℝ := 2.00)
  (vegetables_bags : ℕ := 4)
  (total_bill : ℝ := 18.00)
  (discount : ℝ := 0.10) :
  let known_items_cost := hamburger_meat + crackers + (vegetables_bags * vegetables_unit_price)
  let total_cost_before_discount C := known_items_cost + C
  in 0.90 * (total_cost_before_discount C) = total_bill → C = 3.50 :=
by
  sorry

end cheese_cost_l384_384503


namespace handshakesCountIsCorrect_l384_384197

-- Define the number of gremlins and imps
def numGremlins : ℕ := 30
def numImps : ℕ := 20

-- Define the conditions based on the problem
def handshakesAmongGremlins : ℕ := (numGremlins * (numGremlins - 1)) / 2
def handshakesBetweenImpsAndGremlins : ℕ := numImps * numGremlins

-- Calculate the total handshakes
def totalHandshakes : ℕ := handshakesAmongGremlins + handshakesBetweenImpsAndGremlins

-- Prove that the total number of handshakes equals 1035
theorem handshakesCountIsCorrect : totalHandshakes = 1035 := by
  sorry

end handshakesCountIsCorrect_l384_384197


namespace sum_of_prime_factors_180360_l384_384987

theorem sum_of_prime_factors_180360 : 
  ∑ p in {2, 3, 5, 167}, p = 177 :=
by
  sorry

end sum_of_prime_factors_180360_l384_384987


namespace range_of_h_l384_384643

noncomputable def h (x : ℝ) : ℝ := 3 / (3 + 5 * x^2)

theorem range_of_h :
  (∀ y ∈ set.Ioc 0 1, ∃ x : ℝ, h x = y) ∧ 
  (∃ x : ℤ, set.Ioc (0 : ℝ) 1 = set.Ioc (x : ℝ) 1) ∧ 
  (0 + 1 = 1) :=
by
  sorry

end range_of_h_l384_384643


namespace problem_proof_l384_384522

-- Define points and a condition for line segment PQ and point T
def point (x y : ℝ) := (x, y)

def on_line_segment (P Q T : ℝ × ℝ) : Prop :=
  ∃ (l : ℝ), 0 ≤ l ∧ l ≤ 1 ∧ T = (l * P.1 + (1 - l) * Q.1, l * P.2 + (1 - l) * Q.2)

-- Given values and conditions
def line_eq (x : ℝ) : ℝ := -5/3 * x + 15

def P := point 9 0
def Q := point 0 15

-- Compute the area of triangle POQ
def area_POQ : ℝ := 1/2 * P.1 * Q.2

-- Area condition
def area_condition (T : ℝ × ℝ) : Prop :=
  1/2 * P.1 * T.2 = area_POQ / 4

-- Point T lies on the line y = -5/3 x + 15
def on_line (T : ℝ × ℝ) : Prop :=
  T.2 = line_eq T.1

-- Final statement to prove
def proof_problem : Prop :=
  ∃ (T : ℝ × ℝ), on_line_segment P Q T ∧ area_condition T ∧ on_line T ∧ (T.1 + T.2 = 10.5)

-- Proof statement
theorem problem_proof : proof_problem :=
by
  -- constructing the proof details
  let T : ℝ × ℝ := (6.75, 3.75)
  have h_line_seg : on_line_segment P Q T := sorry
  have h_area : area_condition T := sorry
  have h_on_line : on_line T := sorry
  use T
  use h_line_seg, h_area, h_on_line
  calc
    T.1 + T.2 = 6.75 + 3.75 := by norm_num
          ... = 10.5 := by norm_num

end problem_proof_l384_384522


namespace gcd_optimal_strategy_l384_384555

theorem gcd_optimal_strategy :
  ∃ S_A S_B : ℕ, 
  (∀ n ∈ {i | i ≤ 81}, n ∈ S_A ∨ n ∈ S_B) ∧
  (∀ n m ∈ S_A, n ≠ m) ∧
  (∀ n m ∈ S_B, n ≠ m) ∧
  (∀ x ∈ S_A, ∀ y ∈ S_B, x ≠ y) ∧
  gcd (sum S_A) (sum S_B) = 41 :=
sorry

end gcd_optimal_strategy_l384_384555


namespace sum_of_15_terms_l384_384852

noncomputable def geometric_sequence (a : ℕ → ℝ) :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem sum_of_15_terms 
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_sum1 : a 1 + a 2 + a 3 = 1)
  (h_sum2 : a 4 + a 5 + a 6 = -2) :
  (a 1 + a 2 + a 3) + (a 4 + a 5 + a 6) + (a 7 + a 8 + a 9) +
  (a 10 + a 11 + a 12) + (a 13 + a 14 + a 15) = 11 :=
sorry

end sum_of_15_terms_l384_384852


namespace john_total_fuel_usage_l384_384865

def city_fuel_rate := 6 -- liters per km for city traffic
def highway_fuel_rate := 4 -- liters per km for highway traffic

def trip1_city_distance := 50 -- km for Trip 1
def trip2_highway_distance := 35 -- km for Trip 2
def trip3_city_distance := 15 -- km for Trip 3 in city traffic
def trip3_highway_distance := 10 -- km for Trip 3 on highway

-- Define the total fuel consumption
def total_fuel_used : Nat :=
  (trip1_city_distance * city_fuel_rate) +
  (trip2_highway_distance * highway_fuel_rate) +
  (trip3_city_distance * city_fuel_rate) +
  (trip3_highway_distance * highway_fuel_rate)

theorem john_total_fuel_usage :
  total_fuel_used = 570 :=
by
  sorry

end john_total_fuel_usage_l384_384865


namespace percentage_increase_in_length_l384_384953

theorem percentage_increase_in_length
  (L B : ℝ)
  (newB : ℝ := 1.30 * B)
  (x : ℝ)
  (newL : ℝ := L * (1 + x / 100))
  (original_area : ℝ := L * B)
  (new_area : ℝ := 1.43 * original_area)
  (h : newL * newB = new_area) : x = 10 :=
by
  have h1 : new_area = 1.43 * L * B := rfl
  have h2 : newB = 1.30 * B := rfl
  have h3 : newL * newB = (L * (1 + x / 100)) * (1.30 * B) := rfl
  sorry

end percentage_increase_in_length_l384_384953


namespace solve_equation_l384_384927

def equation_holds (x : ℝ) : Prop := 
  (1 / (x + 10)) + (1 / (x + 8)) = (1 / (x + 11)) + (1 / (x + 7))

theorem solve_equation : equation_holds (-9) :=
by
  sorry

end solve_equation_l384_384927


namespace amount_spent_on_food_l384_384109

-- We define the conditions given in the problem
def Mitzi_brought_money : ℕ := 75
def ticket_cost : ℕ := 30
def tshirt_cost : ℕ := 23
def money_left : ℕ := 9

-- Define the total amount Mitzi spent
def total_spent : ℕ := Mitzi_brought_money - money_left

-- Define the combined cost of the ticket and T-shirt
def combined_cost : ℕ := ticket_cost + tshirt_cost

-- The proof goal
theorem amount_spent_on_food : total_spent - combined_cost = 13 := by
  sorry

end amount_spent_on_food_l384_384109


namespace circle_tangent_arith_prog_and_length_relation_l384_384155

theorem circle_tangent_arith_prog_and_length_relation
  (R P A B C D E : Point)
  (r ρ : ℝ)
  (h1 : r > ρ)
  (h2 : circles_touch_externally_at (R, r) (P, ρ) A)
  (h3 : tangent_touches_circle (R, r) B)
  (h4 : tangent_touches_circle (P, ρ) C)
  (h5 : RP_line_meets_circle_again (P, ρ) D)
  (h6 : circle_meets_line RP BC E)
  (h7 : |BC| = 6 * |DE|)
  (h8 : R_distance_RP = 2 * ρ)
  :
  (lengths_in_arith_prog (triangle R B E) )
  ∧
  (|AB| = 2 * |AC|) := by
  sorry

end circle_tangent_arith_prog_and_length_relation_l384_384155


namespace chris_current_age_l384_384499

def praveens_age_after_10_years (P : ℝ) : ℝ := P + 10
def praveens_age_3_years_back (P : ℝ) : ℝ := P - 3

def praveens_age_condition (P : ℝ) : Prop :=
  praveens_age_after_10_years P = 3 * praveens_age_3_years_back P

def chris_age (P : ℝ) : ℝ := (P - 4) - 2

theorem chris_current_age (P : ℝ) (h₁ : praveens_age_condition P) :
  chris_age P = 3.5 :=
sorry

end chris_current_age_l384_384499


namespace positive_integer_solutions_l384_384901

theorem positive_integer_solutions:
  ∀ (x y : ℕ), (5 * x + y = 11) → (x > 0) → (y > 0) → (x = 1 ∧ y = 6) ∨ (x = 2 ∧ y = 1) :=
by
  sorry

end positive_integer_solutions_l384_384901


namespace parabola_directrix_tangent_line_ln_div_x_ellipse_perimeter_function_increasing_range_m_l384_384580

-- Problem (1)
theorem parabola_directrix (a : ℝ) (h : ∃ y, ∀ x, y = a * x^2 ∧ y = 1) : a = -1/4 :=
sorry

-- Problem (2)
theorem tangent_line_ln_div_x (l : ℝ) (h : ∀x: ℝ, x = 1 → y = (real.log x) / x) : l = (fun x => x - 1) :=
sorry

-- Problem (3)
theorem ellipse_perimeter (a b: ℝ) (h : (∃ (x y : ℝ), x^2 / 16 + y^2 / 25 = 1) ∧ ∃ (f₁ f₂ : (ℝ × ℝ)), a = 4 ∧ b = 5) :
  ∀ (A B : ℝ × ℝ), (A.1^2 / 16 + A.2^2 / 25 = 1 ∧ B.1^2 / 16 + B.2^2 / 25 = 1 ∧ ∃ (F₁ F₂ : (ℝ × ℝ)), true) →
  real.abs (A.1 - B.1) + real.abs (A.2 - B.2) = 20 :=
sorry 

-- Problem (4)
theorem function_increasing_range_m (f : ℝ → ℝ) (m : ℝ)
  (h : (∀ x : ℝ, 2 ≤ x → monotone_increasing (f x)) ∧ f = λ x, x^2 - m * real.log x) :
  (-∞ < m ≤ 8) :=
sorry

end parabola_directrix_tangent_line_ln_div_x_ellipse_perimeter_function_increasing_range_m_l384_384580


namespace main_theorem_l384_384265

def f (m: ℕ) : ℕ := m * (m + 1) / 2

lemma f_1 : f 1 = 1 := by 
  -- placeholder for proof
  sorry

lemma f_functional_eq (m n : ℕ) : f m + f n = f (m + n) - m * n := by
  -- placeholder for proof
  sorry

theorem main_theorem (m : ℕ) : f m = m * (m + 1) / 2 := by
  -- Combining the conditions to conclude the result
  sorry

end main_theorem_l384_384265


namespace jeep_initial_distance_l384_384268

theorem jeep_initial_distance (D : ℝ) (h1 : ∀ t : ℝ, t = 4 → D / t = 103.33 * (3 / 8)) :
  D = 275.55 :=
sorry

end jeep_initial_distance_l384_384268


namespace CorrectStatement_l384_384386

-- Definition of conditions
def Statement1 (a b c : ℝ) : Prop := (ac > bc) → (a > b)
def Statement2 (a b c : ℝ) : Prop := (a < b) → (ac² < bc²)
def Statement3 (a b : ℝ) : Prop := (1/a < 1/b ∧ 1/b < 0) → (a > b)
def Statement4 (a b c d : ℝ) : Prop := (a > b ∧ c > d) → (a - c > b - d)
def Statement5 (a b c d : ℝ) : Prop := (a > b ∧ c > d) → (ac > bd)

-- Main theorem
theorem CorrectStatement : 
  (∀ a b c, ¬Statement1 a b c) ∧ 
  (∀ a b c, ¬Statement2 a b c) ∧ 
  (∀ a b, Statement3 a b) ∧ 
  (∀ a b c d, ¬Statement4 a b c d) ∧ 
  (∀ a b c d, ¬Statement5 a b d) := 
by 
  sorry

end CorrectStatement_l384_384386


namespace number_of_two_digit_values_l384_384878

def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

def satisfies_condition (x : ℕ) : Prop :=
  10 ≤ x ∧ x < 100 ∧ sum_of_digits (sum_of_digits x) = 5

theorem number_of_two_digit_values : #{x : ℕ | satisfies_condition x} = 12 := 
  sorry

end number_of_two_digit_values_l384_384878


namespace graph_intersection_l384_384791

noncomputable
def intersection_set (m : ℝ) (x : ℝ) : Prop :=
  x ∈ Icc 0 1 ∧ (mx - 1)^2 = sqrt x + m

theorem graph_intersection (m : ℝ) :
  0 < m → (∃! x : ℝ, intersection_set m x)
  ↔ (m ∈ Icc 0 1 ∨ m ≥ 3) :=
by
  sorry

end graph_intersection_l384_384791


namespace chord_twice_distance_l384_384840

theorem chord_twice_distance {A B C K I : Type*} [triangle A B C]
  (hC : angle A B C = π / 2)
  (inscribed_circle_touch : touches_inscribed_circle B C K)
  (hAK : cuts_chord AK (inscribed_circle_touch)) :
  chord_length (AK) = 2 * distance C (AK) :=
begin
  -- TODO: Proof goes here
  sorry
end

end chord_twice_distance_l384_384840


namespace seashells_count_l384_384133

theorem seashells_count : 18 + 47 = 65 := by
  sorry

end seashells_count_l384_384133


namespace max_t2_eq_8R2_l384_384579

theorem max_t2_eq_8R2 (R : ℝ) (P Q R : Point) 
  (hPQ : (PQ : Segment) = Segment.diameter R) 
  (hPR : P ≠ R) (hQR : Q ≠ R) :
  let PR := distance P R,
      QR := distance Q R,
      t := PR + QR,
      t2 := t * t
  in max t2 = 8 * R^2 := 
sorry

end max_t2_eq_8R2_l384_384579


namespace equivalent_problem_l384_384068

noncomputable def line_l_standard_eq :=
  ∀ t : ℝ, ∃ x y: ℝ, (x = -5 + (Real.sqrt 2 / 2) * t) ∧ (y = -1 + (Real.sqrt 2 / 2) * t) ∧ (y = x + 4)

noncomputable def curve_C_eq :=
  ∀ θ : ℝ, ∃ ρ : ℝ, (ρ = 4 * Real.cos θ) ∧ (ρ^2 = 4 * ρ * Real.cos θ) ∧ (∀ x y: ℝ, (x = ρ * Real.cos θ) ∧ (y = ρ * Real.sin θ) → (x^2 + y^2 = 4 * x))

noncomputable def max_distance_from_P_to_l :=
  ∀ P : ℝ × ℝ, ∃ d : ℝ, (P.1^2 + P.2^2 = 4 * P.1) ∧ (P.2 = P.1 + 4) ∧ (d = (3 * Real.sqrt 2 + 2))

theorem equivalent_problem (t θ : ℝ) (P : ℝ × ℝ) :
  line_l_standard_eq ∧ curve_C_eq ∧ max_distance_from_P_to_l :=
by
  split
  {
    unfold line_l_standard_eq,
    intros,
    sorry
  },
  {
    unfold curve_C_eq,
    intros,
    sorry
  },
  {
    unfold max_distance_from_P_to_l,
    intros,
    sorry
  }

end equivalent_problem_l384_384068


namespace terminating_decimals_l384_384712

theorem terminating_decimals (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 180) :
  (∃ k : ℕ, k = 20 ∧ ∀ m : ℕ, 1 ≤ m ∧ m ≤ 180 → (decimal_terminates (m / 180) ↔ m % 9 = 0)) :=
by {
  sorry
}

def decimal_terminates (x : ℚ) : Prop :=
  sorry

end terminating_decimals_l384_384712


namespace multiplication_results_l384_384630

theorem multiplication_results
  (h1 : 25 * 4 = 100) :
  25 * 8 = 200 ∧ 25 * 12 = 300 ∧ 250 * 40 = 10000 ∧ 25 * 24 = 600 :=
by
  sorry

end multiplication_results_l384_384630


namespace terminating_decimals_count_l384_384718

theorem terminating_decimals_count :
  ∃ (count : ℕ), count = 60 ∧
  ∀ (n : ℕ), 1 ≤ n ∧ n ≤ 180 →
  (∃ m : ℕ, m * 180 = n * (2^2 * 5)) → 60 = count := sorry

end terminating_decimals_count_l384_384718


namespace terminating_decimal_integers_count_l384_384743

/-- For how many integer values of n between 1 and 180 inclusive does the decimal 
representation of n / 180 terminate? We need to calculate the number of integers n such 
that n includes at least 9 as a factor. The answer is 20. -/
theorem terminating_decimal_integers_count :
  { n : ℤ | 1 ≤ n ∧ n ≤ 180 ∧ 9 ∣ n }.to_finset.card = 20 := 
by sorry

end terminating_decimal_integers_count_l384_384743


namespace problem_statement_l384_384409

def h (x : ℝ) : ℝ := 3 * x + 2
def k (x : ℝ) : ℝ := 2 * x - 3

theorem problem_statement : (h (k (h 3))) / (k (h (k 3))) = 59 / 19 := by
  sorry

end problem_statement_l384_384409


namespace maximum_area_of_region_l384_384433

/-- Given four circles with radii 2, 4, 6, and 8, tangent to the same point B 
on a line ℓ, with the two largest circles (radii 6 and 8) on the same side of ℓ,
prove that the maximum possible area of the region consisting of points lying
inside exactly one of these circles is 120π. -/
theorem maximum_area_of_region 
  (radius1 : ℝ) (radius2 : ℝ) (radius3 : ℝ) (radius4 : ℝ)
  (line : ℝ → Prop) (B : ℝ)
  (tangent1 : ∀ x, line x → dist x B = radius1) 
  (tangent2 : ∀ x, line x → dist x B = radius2)
  (tangent3 : ∀ x, line x → dist x B = radius3)
  (tangent4 : ∀ x, line x → dist x B = radius4)
  (side1 : ℕ)
  (side2 : ℕ)
  (equal_side : side1 = side2)
  (r1 : ℝ := 2) 
  (r2 : ℝ := 4)
  (r3 : ℝ := 6) 
  (r4 : ℝ := 8) :
  (π * (radius1 * radius1) + π * (radius2 * radius2) + π * (radius3 * radius3) + π * (radius4 * radius4)) = 120 * π := 
sorry

end maximum_area_of_region_l384_384433


namespace num_of_subsets_of_set_1_2_l384_384534

-- Define the set and the calculation of the number of subsets
theorem num_of_subsets_of_set_1_2 : (2 ^ 2) = 4 :=
by {
  -- Calculate the number of elements in the set {1, 2}
  let n := 2,
  -- Use the formula to find the number of subsets
  have h : 2 ^ n = 2 ^ 2 := by rfl,
  -- Conclude that 2 ^ 2 = 4
  exact h,
  sorry
}

end num_of_subsets_of_set_1_2_l384_384534


namespace max_harmonic_pairs_l384_384345

def Point := (ℝ × ℝ)

def manhattan_distance (A B : Point) : ℝ :=
  (abs (A.1 - B.1)) + (abs (A.2 - B.2))

def harmonic (A B : Point) : Prop :=
  1 < manhattan_distance A B ∧ manhattan_distance A B ≤ 2

theorem max_harmonic_pairs (pts : Finset Point) (h : pts.card = 100) :
  ∃ n, n = 3750 ∧ ∀ (A B : Point), A ∈ pts → B ∈ pts → harmonic A B → (A, B) ∈ (Finset.product pts pts).filter (λ (p : Point × Point), harmonic p.1 p.2) :=
sorry

end max_harmonic_pairs_l384_384345


namespace race_distance_l384_384835

theorem race_distance {d x y z : ℝ} :
  (d / x = (d - 25) / y) →
  (d / y = (d - 15) / z) →
  (d / x = (d - 37) / z) →
  d = 125 :=
by
  intros h1 h2 h3
  -- Insert proof here
  sorry

end race_distance_l384_384835


namespace question1_question2_l384_384030

section

variable (A B C : Set ℝ)
variable (a : ℝ)

-- Condition 1: A = {x | -1 ≤ x < 3}
def setA : Set ℝ := {x | -1 ≤ x ∧ x < 3}

-- Condition 2: B = {x | 2x - 4 ≥ x - 2}
def setB : Set ℝ := {x | x ≥ 2}

-- Condition 3: C = {x | x ≥ a - 1}
def setC (a : ℝ) : Set ℝ := {x | x ≥ a - 1}

-- Question 1: Prove A ∩ B = {x | 2 ≤ x < 3}
theorem question1 : A = setA → B = setB → A ∩ B = {x | 2 ≤ x ∧ x < 3} :=
by intros hA hB; rw [hA, hB]; sorry

-- Question 2: If B ∪ C = C, prove a ∈ (-∞, 3]
theorem question2 : B = setB → C = setC a → (B ∪ C = C) → a ≤ 3 :=
by intros hB hC hBUC; rw [hB, hC] at hBUC; sorry

end

end question1_question2_l384_384030


namespace sin_sum_diff_l384_384353

theorem sin_sum_diff (α β : ℝ) 
  (hα : Real.sin α = 1/3) 
  (hβ : Real.sin β = 1/2) : 
  Real.sin (α + β) * Real.sin (α - β) = -5/36 := 
sorry

end sin_sum_diff_l384_384353


namespace distance_sum_ratio_l384_384873

-- Define the regular tetrahedron structure, the point inside it, and distance functions

structure Tetrahedron :=
  (A B C D : Point)

structure InnerPoint (T : Tetrahedron) :=
  (E : Point)

def distance_to_face (E : Point) (face : Face) : ℝ := sorry -- hypothetical function

def distance_to_edge (E : Point) (edge : Edge) : ℝ := sorry -- hypothetical function

def sum_of_distances_to_faces (T : Tetrahedron) (E : Point) : ℝ :=
  distance_to_face E T.ABC + distance_to_face E T.ABD + 
  distance_to_face E T.ACD + distance_to_face E T.BCD

def sum_of_distances_to_edges (T : Tetrahedron) (E : Point) : ℝ :=
  distance_to_edge E T.AB + distance_to_edge E T.AC + 
  distance_to_edge E T.AD + distance_to_edge E T.BC + 
  distance_to_edge E T.BD + distance_to_edge E T.CD

theorem distance_sum_ratio (T : Tetrahedron) (P : InnerPoint T) :
  (sum_of_distances_to_faces T P.E) / (sum_of_distances_to_edges T P.E) = 1 / 2 :=
by 
  sorry

end distance_sum_ratio_l384_384873


namespace cuboid_on_sphere_surface_area_l384_384360

noncomputable def cuboid_diagonal (a : ℝ) : ℝ :=
  real.sqrt (4 * a^2 + a^2 + a^2)

noncomputable def sphere_surface_area (R : ℝ) : ℝ :=
  4 * real.pi * R^2

theorem cuboid_on_sphere_surface_area (a : ℝ) (h : 0 < a) :
    sphere_surface_area (cuboid_diagonal a / 2) = 6 * real.pi * a^2 :=
by
  sorry

end cuboid_on_sphere_surface_area_l384_384360


namespace length_of_PS_l384_384435

theorem length_of_PS
  (PT TR QT TS PQ : ℝ)
  (h1 : PT = 5)
  (h2 : TR = 7)
  (h3 : QT = 9)
  (h4 : TS = 4)
  (h5 : PQ = 7) :
  PS = Real.sqrt 66.33 := 
  sorry

end length_of_PS_l384_384435


namespace find_n_l384_384461

noncomputable def problem_statement (r : ℝ) (n : ℕ) :=
  (∀ (a : ℕ → ℝ), (∀ i, 0 < a i) →
    (finset.univ.sum (λ i, a i) = r * (finset.univ.sum (λ i, 1 / a i))) →
    (finset.univ.sum (λ i, 1 / (real.sqrt r - a i)) = 1 / real.sqrt r))

theorem find_n (r : ℝ) (hr : 0 < r) (hn : problem_statement r n) : n = 2 := sorry

end find_n_l384_384461


namespace initial_books_in_bin_l384_384586

variable (X : ℕ)

theorem initial_books_in_bin (h1 : X - 3 + 10 = 11) : X = 4 :=
by
  sorry

end initial_books_in_bin_l384_384586


namespace repairman_probability_l384_384277

-- Define the arrival time domain and the uniform distribution
def arrival_time_domain : set ℝ := {x | 10 ≤ x ∧ x ≤ 18}
def client_away_interval : set ℝ := {x | 14 ≤ x ∧ x < 15}

noncomputable def uniform_pdf (a b : ℝ) : ℝ → ℝ := 
  λ x, if x ∈ Icc a b then 1 / (b - a) else 0

-- Define the probability function over the given domain
noncomputable def probability (interval : set ℝ) (a b : ℝ) : ℝ :=
  ∫ x in interval, uniform_pdf a b x

-- State the proof problem in Lean 4
theorem repairman_probability : probability client_away_interval 14 18 = 1 / 4 :=
by
  -- proof goes here
  sorry

end repairman_probability_l384_384277


namespace kyle_delivers_daily_papers_l384_384457

theorem kyle_delivers_daily_papers (x : ℕ) (h : 6 * x + (x - 10) + 30 = 720) : x = 100 :=
by
  sorry

end kyle_delivers_daily_papers_l384_384457


namespace triangle_DBC_is_equilateral_l384_384083

open EuclideanGeometry

variables {A B C D : Point}
variables {a b c ad ab ac : ℝ}

-- Given conditions

axiom angle_A (hABC : Triangle A B C) : angle A = 120
axiom D_on_angle_bisector (hABC : Triangle A B C) (hD : IsBisector D A angle A)
axiom length_AD_eq_ab_ac (hAD : Distance A D) (hAB : Distance A B) (hAC : Distance A C) : Distance A D = Distance A B + Distance A C

-- To Prove

theorem triangle_DBC_is_equilateral (hABC : Triangle A B C) (hD : IsBisector D A angle A) (hAD : Distance A D) (hAB : Distance A B) (hAC : Distance A C) 
  (ha : Distance A D = Distance A B + Distance A C) (hb : angle A = 120) : Equilateral (Triangle D B C) := 
  sorry

end triangle_DBC_is_equilateral_l384_384083


namespace green_ducks_percentage_l384_384432

noncomputable def percentage_of_green_ducks_in_larger_pond : ℕ :=
  let smaller_pond_ducks := 20
  let larger_pond_ducks := 80
  let green_ducks_in_smaller_pond := 0.20 * smaller_pond_ducks
  let total_ducks := smaller_pond_ducks + larger_pond_ducks
  let green_percentage_total := 0.16
  let total_green_ducks := green_percentage_total * total_ducks
  let green_ducks_in_larger_pond := total_green_ducks - green_ducks_in_smaller_pond
  let green_ducks_perc_in_larger := (green_ducks_in_larger_pond / larger_pond_ducks) * 100
  green_ducks_perc_in_larger

theorem green_ducks_percentage : percentage_of_green_ducks_in_larger_pond = 15 := 
by
  sorry

end green_ducks_percentage_l384_384432


namespace arrange_from_smallest_to_largest_l384_384294

noncomputable def log (a b : ℝ) := Real.log b / Real.log a
noncomputable def exp (base x : ℝ) := base^x

variable (base1 base2 x1 x2 : ℝ)

theorem arrange_from_smallest_to_largest :
  0.2 > 0 ∧ 2.3 > 0 ∧ log 0.2 2.3 < 0 ∧ 0.2^(-0.2) > 0 ∧ 2.3^(-2.3) > 0 ∧ 
  (-0.2 < 0) ∧ (0.2^(-0.2) > 1) ∧ 
  (-2.3 < 0) ∧ (2.3^(-2.3) < 1) →
  log 0.2 2.3 < 2.3^(-2.3) ∧ 2.3^(-2.3) < 0.2^(-0.2) :=
by
  intros
  sorry

end arrange_from_smallest_to_largest_l384_384294


namespace ellipse_slope_product_constant_l384_384773

theorem ellipse_slope_product_constant (k b : ℝ) (hkb : k ≠ 0 ∧ b ≠ 0) :
  (∃ (x y : ℝ), (x, y) ∈ set_of (λ p : ℝ × ℝ, (p.1^2 / 8 + p.2^2 / 4 = 1)) ∧
  (∃ x1 x2 y1 y2 : ℝ, (x1, y1) ≠ (x2, y2) ∧ y1 = k * x1 + b ∧ y2 = k * x2 + b ∧
  let xM := (x1 + x2) / 2 in let yM := k * xM + b in
  (yM / xM) * k = -1 / 2))
:= sorry

end ellipse_slope_product_constant_l384_384773


namespace grid_shaded_area_l384_384060

theorem grid_shaded_area :
  let grid_side := 12
  let grid_area := grid_side^2
  let radius_small := 1.5
  let radius_large := 3
  let area_small := π * radius_small^2
  let area_large := π * radius_large^2
  let total_area_circles := 3 * area_small + area_large
  let visible_area := grid_area - total_area_circles
  let A := 144
  let B := 15.75
  A = 144 ∧ B = 15.75 ∧ (A + B = 159.75) →
  visible_area = 144 - 15.75 * π :=
by
  intros
  sorry

end grid_shaded_area_l384_384060


namespace find_a_l384_384762

def f (a x : ℝ) : ℝ :=
  if x ≤ a then sin (2 * Real.pi * (x - a)) else x ^ 2 - 2 * (a + 1) * x + a ^ 2

theorem find_a (a : ℝ) (h : a ∈ Icc (-1:ℝ) 1) : f a (f a a) = 1 ↔ a = -1 ∨ a = 3 / 4 :=
by
  sorry

end find_a_l384_384762


namespace best_fitting_model_is_model1_l384_384854

def model (n : ℕ) := ℕ → ℝ -- assume models are indexed as natural numbers with coefficients as reals.

def R_squared : ℕ → ℝ
| 1 := 0.98
| 2 := 0.80
| 3 := 0.50
| 4 := 0.25
| _ := 0  -- other cases default to 0 for simplicity

theorem best_fitting_model_is_model1 : ∀ m, 1 ≤ m ∧ m ≤ 4 → R_squared m ≤ R_squared 1 :=
begin
  intros m hm,
  cases m,
  { linarith }, -- not valid since m must be between 1 and 4.
  cases m,
  { linarith }, -- when m = 1, this is true by definition.
  cases m,
  { exact le_of_lt (by norm_num) }, -- m = 2, R^2 is 0.8 which is less than 0.98
  cases m,
  { exact le_of_lt (by norm_num) }, -- m = 3, R^2 is 0.5 which is less than 0.98
  cases m,
  { exact le_of_lt (by norm_num) }, -- m = 4, R^2 is 0.25 which is less than 0.98
  cases m,
  { linarith }, -- (m > 4 is defaulted to R_squared = 0, hence less than 0.98)
end

end best_fitting_model_is_model1_l384_384854


namespace sand_height_when_inverted_l384_384258

noncomputable def sand_height_inverted (r h hf: ℝ) (h_cylinder: ℝ) : ℝ :=
  let volume_cone := (1 / 3) * real.pi * r * r * h,
      volume_cylinder_part := real.pi * r * r * h_cylinder,
      total_volume := volume_cone + volume_cylinder_part,
      volume_new_cylinder := total_volume - volume_cone,
      new_height := volume_new_cylinder / (real.pi * r * r)
  in (h + new_height)

theorem sand_height_when_inverted (r h hf : ℝ) (h_cylinder: ℝ) : 
  r = 12 → h = 20 → hf = 20 → h_cylinder = 5 → 
  sand_height_inverted r h hf h_cylinder = 25 :=
by { intros, unfold sand_height_inverted, sorry }

end sand_height_when_inverted_l384_384258


namespace equal_share_payment_l384_384090

theorem equal_share_payment (A B C : ℝ) (h : A < B ∧ B < C) :
  (B + C - 2 * A) / 3 + (A + B - 2 * C) / 3 = ((A + B + C) * 2 / 3) - B :=
by
  sorry

end equal_share_payment_l384_384090


namespace polynomial_zero_existence_l384_384305

theorem polynomial_zero_existence (f g : ℝ → ℝ) (a : Fin n → ℝ) (n : ℕ)
  (h_f : ∀ x, f x = ∑ k in Finset.range n, a k * x^(k + 1))
  (h_g : ∀ x, g x = ∑ k in Finset.range n, a k / (2^(k + 1) - 1) * x^(k + 1))
  (h_g1 : g 1 = 0) (h_g2_n1 : g (2^(n + 1)) = 0) : 
  ∃ x ∈ Ioo (0 : ℝ) (2^n : ℝ), f x = 0 := 
sorry

end polynomial_zero_existence_l384_384305


namespace gcd_8251_6105_l384_384210

theorem gcd_8251_6105 : Nat.gcd 8251 6105 = 37 := by
  sorry

end gcd_8251_6105_l384_384210


namespace days_to_navigate_from_shanghai_to_vancouver_l384_384587

noncomputable def navigation_days_from_shanghai_to_vancouver : ℕ :=
let customs_days := 4 in
let transport_days := 7 in
let expected_days := 2 in
let departed_days_ago := 30 in
let total_days_expected := departed_days_ago + expected_days in
let total_days := customs_days + transport_days in
total_days_expected - total_days

theorem days_to_navigate_from_shanghai_to_vancouver 
(h_customs: customs_days = 4)
(h_transport: transport_days = 7)
(h_expected: expected_days = 2)
(h_departed: departed_days_ago = 30) :
  navigation_days_from_shanghai_to_vancouver = 21 := by {
  sorry
}

end days_to_navigate_from_shanghai_to_vancouver_l384_384587


namespace mona_unique_players_l384_384110

variable (G : Type)  -- Type of the group of players
variable (mona : G)
variable (groups_in_weekend : List (List G))  -- List of the groups Mona joined over the weekend

-- Conditions
def num_groups : Nat := 18
def groups_with_two_repeats : List (List G) := [g1, g2, g3, g4, g5, g6]  -- 6 groups with 2 repeated players each
def groups_with_one_repeat : List (List G) := [g7, g8, g9, g10]  -- 4 groups with 1 repeated player each
def large_scale_group_1 : List G := g11  -- Group with 9 players and 4 repeated players
def large_scale_group_2 : List G := g12  -- Group with 12 players and 5 repeated players

axiom group_size (g : List G) : Nat
axiom repeats_in_group (g : List G) : Nat  -- Number of repeated players in a group

def total_repeats : Nat := 
  (group_size g1 + group_size g2 + group_size g3 + group_size g4 + group_size g5 + group_size g6) +
  (repeats_in_group g1 + repeats_in_group g2 + repeats_in_group g3 + repeats_in_group g4 + repeats_in_group g5 + repeats_in_group g6) + 
  (group_size g7 + group_size g8 + group_size g9 + group_size g10) * 1 +
  (repeats_in_group g7 + repeats_in_group g8 + repeats_in_group g9 + repeats_in_group g10) 

def total_unique_players_assumption: Nat :=
  (num_groups - (List.length groups_with_two_repeats + List.length groups_with_one_repeat)) +
  (group_size large_scale_group_1 - repeats_in_group large_scale_group_1) +
  (group_size large_scale_group_2 - repeats_in_group large_scale_group_2)

theorem mona_unique_players : 
  total_unique_players_assumption G mona groups_in_weekend groups_with_two_repeats groups_with_one_repeat large_scale_group_1 large_scale_group_2 = 20 := 
by
  sorry

end mona_unique_players_l384_384110


namespace math_equivalent_problem_l384_384063

noncomputable def correct_difference (A B C D : ℕ) (incorrect_difference : ℕ) : ℕ :=
  if (B = 3) ∧ (D = 2) ∧ (C = 5) ∧ (incorrect_difference = 60) then
    ((A * 10 + B) - 52)
  else
    0

theorem math_equivalent_problem (A : ℕ) : correct_difference A 3 5 2 60 = 31 :=
by
  sorry

end math_equivalent_problem_l384_384063


namespace l_shape_coverage_l384_384984

-- Definitions based on the conditions
def is_covered_by_l_shape {n : ℕ} (k : ℕ) (marked_cells : Finset (Fin n × Fin n)) : Prop :=
  ∀ (l_shape : Finset (Fin n × Fin n)), l_shape.card = 3 → 
  (∃ (c1 c2 : Fin n × Fin n), c1 ∈ marked_cells ∧ c2 ∈ marked_cells ∧ c1 ≠ c2 ∧ c1 ∈ l_shape ∧ c2 ∈ l_shape)

-- The board is specified as 9x9 and k as the minimal marking cells
def smallest_k_to_cover_l_shape : ℕ :=
  56

-- The statement we want to prove
theorem l_shape_coverage :
  ∃ (k : ℕ) (marked_cells : Finset (Fin 9 × Fin 9)),
    k = smallest_k_to_cover_l_shape ∧ is_covered_by_l_shape 9 k marked_cells :=
by
  let k := smallest_k_to_cover_l_shape
  sorry

end l_shape_coverage_l384_384984


namespace find_a_value_l384_384796

theorem find_a_value (x a : ℝ) (h1 : x > 0) (h2 : a > 0) (h3 : ∀ y > 0, (4 * x + a / x ≤ 4 * y + a / y)) : a = 36 :=
by
  have f'_x := deriv (λ x, 4 * x + a / x) x
  simp at f'_x
  sorry

end find_a_value_l384_384796


namespace group_purchase_cheaper_l384_384237

-- Define the initial conditions
def initial_price : ℕ := 10
def bulk_price : ℕ := 7
def delivery_cost : ℕ := 100
def group_size : ℕ := 50

-- Define the costs for individual and group purchases
def individual_cost : ℕ := initial_price
def group_cost : ℕ := bulk_price + (delivery_cost / group_size)

-- Statement to prove: cost per participant in a group purchase is less than cost per participant in individual purchases
theorem group_purchase_cheaper : group_cost < individual_cost := by
  sorry

end group_purchase_cheaper_l384_384237


namespace people_got_off_at_first_stop_l384_384545

theorem people_got_off_at_first_stop 
  (X : ℕ)
  (h1 : 50 - X - 6 - 1 = 28) :
  X = 15 :=
by
  sorry

end people_got_off_at_first_stop_l384_384545


namespace tangent_line_equation_at_point_l384_384160

theorem tangent_line_equation_at_point :
  let y := λ x : ℝ, (1 + x) / (1 - x)
  let x₀ := 2
  let y₀ := -3
  let m := 2
  (2 * x₀ - y₀ - 7 = 0) :=
by 
  sorry

end tangent_line_equation_at_point_l384_384160


namespace circle_minus_sector_area_l384_384982

variable (P Q : ℝ × ℝ)
variable (θ : ℝ)

def radius (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((Q.1 - P.1) ^ 2 + (Q.2 - P.2) ^ 2)

def circle_area (r : ℝ) : ℝ :=
  real.pi * r ^ 2

def sector_area (r θ : ℝ) : ℝ :=
  1 / 2 * r ^ 2 * θ

theorem circle_minus_sector_area (hP : P = (1, 2)) (hQ : Q = (7, 6)) (hθ : θ = real.pi / 3) :
    let r := radius P Q in
    let A := circle_area r in
    let A_sector := sector_area r θ in
    A - A_sector = (130 * real.pi) / 3 := by
  sorry

end circle_minus_sector_area_l384_384982


namespace perpendicular_tangent_line_a_value_l384_384018

theorem perpendicular_tangent_line_a_value :
  ∀ (a : ℝ),
    (∀ (x y : ℝ), y = x^3 + 2 * x → (x = 1 ∧ y = 3) →
     ∀ (m : ℝ), m = 3 * x ^ 2 + 2 →
       (∀ u v : ℝ, v = m → 
        (a * u - v + 2019 = 0 → 
         m * a = -1))) →
  a = -1 / 5 :=
begin
  sorry
end

end perpendicular_tangent_line_a_value_l384_384018


namespace f_of_3_unique_value_l384_384880

noncomputable def f : ℝ → ℝ := sorry

-- The main statement to be proven
theorem f_of_3_unique_value:
  (∀ x y : ℝ, f(x * f(y) + 2 * x) = 2 * x * y + f(x)) →
  (∃! z : ℝ, f 3 = z) ∧ z = -2 :=
by 
  sorry

end f_of_3_unique_value_l384_384880


namespace two_rooks_non_capture_count_two_kings_non_capture_count_two_bishops_non_capture_count_two_knights_non_capture_count_two_queens_non_capture_count_l384_384065

-- Two Rooks
theorem two_rooks_non_capture_count : 
  let valid_ways := 1568
  in valid_ways = number_of_non_capturing_placements "rook" 8 :=
by sorry

-- Two Kings
theorem two_kings_non_capture_count : 
  let valid_ways := 1806
  in valid_ways = number_of_non_capturing_placements "king" 8 :=
by sorry

-- Two Bishops
theorem two_bishops_non_capture_count : 
  let valid_ways := 1972
  in valid_ways = number_of_non_capturing_placements "bishop" 8 :=
by sorry

-- Two Knights
theorem two_knights_non_capture_count : 
  let valid_ways := 1848
  in valid_ways = number_of_non_capturing_placements "knight" 8 :=
by sorry

-- Two Queens
theorem two_queens_non_capture_count : 
  let valid_ways := 1980
  in valid_ways = number_of_non_capturing_placements "queen" 8 :=
by sorry

end two_rooks_non_capture_count_two_kings_non_capture_count_two_bishops_non_capture_count_two_knights_non_capture_count_two_queens_non_capture_count_l384_384065


namespace max_acceptable_ages_l384_384937

noncomputable def acceptable_ages (avg_age std_dev : ℕ) : ℕ :=
  let lower_limit := avg_age - 2 * std_dev
  let upper_limit := avg_age + 2 * std_dev
  upper_limit - lower_limit + 1

theorem max_acceptable_ages : acceptable_ages 40 10 = 41 :=
by
  sorry

end max_acceptable_ages_l384_384937


namespace slope_angle_at_point_l384_384186

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 4 * x + 8

-- Define the derivative of the function f
def f' (x : ℝ) : ℝ := 3 * x^2 - 4

-- State the problem: Prove the slope angle at point (1, 5) is 135 degrees
theorem slope_angle_at_point (θ : ℝ) (h : θ = 135) :
    f' 1 = -1 := 
by 
    sorry

end slope_angle_at_point_l384_384186


namespace max_of_linear_trig_combination_l384_384879

variables (a b φ θ : ℝ)

theorem max_of_linear_trig_combination :
  ∃ (θ : ℝ), a * cos(θ - φ) + b * sin(θ - φ) ≤ sqrt(a^2 + b^2) :=
sorry

end max_of_linear_trig_combination_l384_384879


namespace amelia_divisions_to_1_l384_384146

def divide_and_floor (n : Nat) : Nat :=
  n / 3

theorem amelia_divisions_to_1 :
  let rec steps_to_reach_1 (n : Nat) (count : Nat) : Nat :=
    if n ≤ 1 then count
    else steps_to_reach_1 (divide_and_floor n) (count + 1)
  in steps_to_reach_1 200 0 = 5 :=
by
  -- placeholder for the proof
  sorry

end amelia_divisions_to_1_l384_384146


namespace gh_commutes_l384_384870

theorem gh_commutes (G : Type) [group G] (m n : ℕ)
  (h_mn : m > 0 ∧ n > 0 ∧ nat.gcd m n = 1)
  (a : fin n → ℕ) (h_a : ∀ k : fin n, a k = (⌊(m * (k : ℕ) + 1) / n⌋ - ⌊m * (k : ℕ) / n⌋))
  (g h : G) (h_gh : list.prod (list.of_fn (λ k, g * h ^ (a k))) = 1) :
  g * h = h * g := 
sorry

end gh_commutes_l384_384870


namespace smallest_ratio_l384_384144

-- Define the system of equations as conditions
def eq1 (x y : ℝ) := x^3 + 3 * y^3 = 11
def eq2 (x y : ℝ) := (x^2 * y) + (x * y^2) = 6

-- Define the goal: proving the smallest value of x/y for the solutions (x, y) is -1.31
theorem smallest_ratio (x y : ℝ) (h1 : eq1 x y) (h2 : eq2 x y) :
  ∃ t : ℝ, t = x / y ∧ ∀ t', t' = x / y → t' ≥ -1.31 :=
sorry

end smallest_ratio_l384_384144


namespace sum_of_exponents_2023_l384_384105

theorem sum_of_exponents_2023 : 
  ∃ (s : List ℕ), (∀ n ∈ s, ∃ k : ℕ, n = 2^k) ∧ (s.sum = 2023) ∧ (s.map (λ n, s.findIndex (λ k, n = 2^k))).sum = 48 := sorry

end sum_of_exponents_2023_l384_384105


namespace two_percent_of_one_l384_384959

-- Definitions of concepts used in the problem
def percent (n : ℝ) : ℝ := n / 100

def percentage_of (percent : ℝ) (number : ℝ) : ℝ := percent * number

-- Statement of the proof problem
theorem two_percent_of_one : percentage_of (percent 2) 1 = (2 / 100) :=
by
  sorry

end two_percent_of_one_l384_384959


namespace number_of_solutions_l384_384402

def satisfies_equation (x : ℝ) : Prop := 
  cos x * cos x + 2 * sin x * sin x = 1

theorem number_of_solutions : 
  ∃ (n : ℕ), n = 43 ∧ (finset.range 39).filter (λ k, -15 < (k - 4) * π ∧ (k - 4) * π < 120) = n :=
sorry

end number_of_solutions_l384_384402


namespace quadratic_two_distinct_real_roots_l384_384183

theorem quadratic_two_distinct_real_roots : 
  ∀ x : ℝ, ∃ a b c : ℝ, (∀ x : ℝ, (x+1)*(x-1) = 2*x + 3 → x^2 - 2*x - 4 = 0) ∧ 
  (a = 1) ∧ (b = -2) ∧ (c = -4) ∧ (b^2 - 4*a*c > 0) :=
by
  sorry

end quadratic_two_distinct_real_roots_l384_384183


namespace horse_revolutions_l384_384267

theorem horse_revolutions (r1 r2 : ℝ) (rev1 rev2 : ℕ) (h1 : r1 = 30) (h2 : rev1 = 25) (h3 : r2 = 10) : 
  rev2 = 75 :=
by 
  sorry

end horse_revolutions_l384_384267


namespace shop_owner_profit_l384_384279

noncomputable def profitA : ℝ := (1.20 - 0.88) / 0.88 * 100
noncomputable def profitB : ℝ := (1.30 - 0.82) / 0.82 * 100
noncomputable def profitC : ℝ := (1.40 - 0.80) / 0.80 * 100

noncomputable def wA : ℝ := 0.30
noncomputable def wB : ℝ := 0.45
noncomputable def wC : ℝ := 0.25

theorem shop_owner_profit :
  let overall_profit := (wA * profitA) + (wB * profitB) + (wC * profitC)
  in overall_profit = 56 := 
by {
  let overall_profit := (wA * profitA) + (wB * profitB) + (wC * profitC)
  sorry
}

end shop_owner_profit_l384_384279


namespace triangle_AF_eq_AB_minus_AC_l384_384011

open Classical

noncomputable theory

variables {A B C E F : Type} [metric_space A] [metric_space B] [metric_space C] [metric_space E] [metric_space F]
variables [inner_product_space ℝ A] [inner_product_space ℝ B] [inner_product_space ℝ C] [inner_product_space ℝ E] [inner_product_space ℝ F]

variables {triangle : Type} [is_triangle triangle A B C]
variables {circumcircle : Type} [is_circumcircle circumcircle A B C]
variables {external_angle_bisector : Type} [is_external_angle_bisector external_angle_bisector A]

-- The primary statement we need to prove
theorem triangle_AF_eq_AB_minus_AC (hAB_gt_AC : AB > AC) 
  (h_external_angle_bisector : external_angle_bisector_intersects_circumcircle_at_E external_angle_bisector circumcircle E)
  (h_perpendicular : ∃ F, is_foot_of_perpendicular F E AB) : 2 * (distance A F) = (distance A B) - (distance A C) :=
by
  have hF : F = classical.some h_perpendicular,
  sorry

end triangle_AF_eq_AB_minus_AC_l384_384011


namespace sum_of_digits_l384_384758

theorem sum_of_digits (A B C D : ℕ) (H1: A < B) (H2: B < C) (H3: C < D)
  (H4: A > 0) (H5: B > 0) (H6: C > 0) (H7: D > 0)
  (H8: 1000 * A + 100 * B + 10 * C + D + 1000 * D + 100 * C + 10 * B + A = 11990) : 
  (A, B, C, D) = (1, 9, 9, 9) :=
sorry

end sum_of_digits_l384_384758


namespace terminating_decimals_count_l384_384730

theorem terminating_decimals_count : 
  let N := 180 in
  let count_terminating_decimals := (List.range' 1 N).filter (λ n, n % 3 = 0) in
  count_terminating_decimals.length = 60 :=
by
  let N := 180
  let count_terminating_decimals := (List.range' 1 N).filter (λ n, n % 3 = 0)
  sorry

end terminating_decimals_count_l384_384730


namespace terminating_decimals_count_l384_384729

theorem terminating_decimals_count : 
  let N := 180 in
  let count_terminating_decimals := (List.range' 1 N).filter (λ n, n % 3 = 0) in
  count_terminating_decimals.length = 60 :=
by
  let N := 180
  let count_terminating_decimals := (List.range' 1 N).filter (λ n, n % 3 = 0)
  sorry

end terminating_decimals_count_l384_384729


namespace area_of_triangle_KMB_l384_384841

-- Definitions based on conditions
def angle_B := Real.arctan (8 / 15)
def radius := 1
def MB := 15 / 8

-- Main statement to prove
theorem area_of_triangle_KMB 
  (h_iso : IsoscelesTriangle ABC)
  (h_angleB : ∠ B = angle_B)
  (h_radius : InscribedCircle radius ∠ C)
  (h_touch : InscribedCircleTouch h_radius CB M)
  (h_points : OrderedPointsOnBase [A, K, E, B] AB)
  (h_MB : MB = 15 / 8) :
  AreaOfTriangle KMB = 375 / 272 :=
  sorry

end area_of_triangle_KMB_l384_384841


namespace age_difference_l384_384280

theorem age_difference {son_age dad_age : ℕ} (h_son : son_age = 9) (h_dad : dad_age = 36) (h_relation : dad_age = 4 * son_age) : dad_age - son_age = 27 :=
by
  rw [h_son, h_dad]
  sorry

end age_difference_l384_384280


namespace find_m_l384_384176

noncomputable def par_tangent_hyp (m : ℝ) : Prop :=
  let y := λ x : ℝ, x^2 + 4 in
  let hyp := λ x y : ℝ, y^2 - m * x^2 = 4 in
  ∃ y x : ℝ, y = x^2 + 4 ∧ y^2 - m * x^2 = 4

theorem find_m : 
  ∃ m : ℝ, (m = 8 + 4 * Real.sqrt 3 ∨ m = 8 - 4 * Real.sqrt 3) ∧ par_tangent_hyp m :=
begin
  sorry
end

end find_m_l384_384176


namespace sin_double_angle_condition_l384_384350

theorem sin_double_angle_condition (θ : ℝ) (h : Real.sin θ + Real.cos θ = 1 / 3) : Real.sin (2 * θ) = -8 / 9 := 
sorry

end sin_double_angle_condition_l384_384350


namespace terminating_decimal_integers_count_l384_384741

/-- For how many integer values of n between 1 and 180 inclusive does the decimal 
representation of n / 180 terminate? We need to calculate the number of integers n such 
that n includes at least 9 as a factor. The answer is 20. -/
theorem terminating_decimal_integers_count :
  { n : ℤ | 1 ≤ n ∧ n ≤ 180 ∧ 9 ∣ n }.to_finset.card = 20 := 
by sorry

end terminating_decimal_integers_count_l384_384741


namespace no_common_points_l384_384803

noncomputable def f (a x : ℝ) : ℝ := x^2 - a * x
noncomputable def g (a b x : ℝ) : ℝ := b + a * Real.log (x - 1)
noncomputable def h (a x : ℝ) : ℝ := x^2 - a * x - a * Real.log (x - 1)
noncomputable def G (a : ℝ) : ℝ := -a^2 / 4 + 1 - a * Real.log (a / 2)

theorem no_common_points (a b : ℝ) (h1 : 1 ≤ a) :
  (∀ x > 1, f a x ≠ g a b x) ↔ b < 3 / 4 + Real.log 2 :=
by
  sorry

end no_common_points_l384_384803


namespace custom_op_evaluation_l384_384046

-- Define the custom operation *
def custom_op (A B : ℝ) : ℝ := (A + B) / 3

-- Proof problem statement
theorem custom_op_evaluation :
  custom_op (custom_op (custom_op 7 11) 5) 13 = 50 / 9 :=
by
  sorry

end custom_op_evaluation_l384_384046


namespace range_of_a_l384_384387

def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then exp (a * x) + 1 else 2 * x^3 - 3 * x^2 + 1

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 2 → f a x ≤ 5) → (a ≥ -ln 2) :=
by
  intro h
  sorry

end range_of_a_l384_384387


namespace x_add_y_add_one_is_composite_l384_384956

theorem x_add_y_add_one_is_composite (x y : ℕ) (hx : x > 1) (hy : y > 1) 
  (k : ℕ) (h : x^2 + x * y - y = k^2) : ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ (x + y + 1 = a * b) :=
by
  sorry

end x_add_y_add_one_is_composite_l384_384956


namespace height_relationship_l384_384556

theorem height_relationship (r1 r2 h1 h2 : ℝ) (h_radii : r2 = 1.2 * r1) (h_volumes : π * r1^2 * h1 = π * r2^2 * h2) : h1 = 1.44 * h2 :=
by
  sorry

end height_relationship_l384_384556


namespace tangent_line_at_2_neg3_l384_384159

noncomputable def tangent_line (x : ℝ) : ℝ := (1 + x) / (1 - x)

theorem tangent_line_at_2_neg3 :
  ∃ m b, ∀ x, (tangent_line x = m * x + b) →
  ∃ y, (2 * x - y - 7 = 0) :=
by
  sorry

end tangent_line_at_2_neg3_l384_384159


namespace terminating_decimals_count_l384_384715

theorem terminating_decimals_count :
  ∃ (count : ℕ), count = 60 ∧
  ∀ (n : ℕ), 1 ≤ n ∧ n ≤ 180 →
  (∃ m : ℕ, m * 180 = n * (2^2 * 5)) → 60 = count := sorry

end terminating_decimals_count_l384_384715


namespace quadratic_solution_unique_l384_384380

theorem quadratic_solution_unique {a c : ℝ} (h1 : a + c = 36) (h2 : a < c) :
  (a ≠ 0) ∧ (15^2 - 4 * a * c = 0) → 
  (a, c) = (36 - Real.sqrt 1071) / 2, (36 + Real.sqrt 1071) / 2 := sorry

end quadratic_solution_unique_l384_384380


namespace apple_costs_l384_384293

-- Define the cost rates for apples
def rate1 : ℕ := 15
def rate2 : ℕ := 25
def pack1 : ℕ := 4
def pack2 : ℕ := 7

-- Define the number of apples and the target costs
def total_apples : ℕ := 25
def cost_25_apples : ℕ := 90
def num_apples_6 : ℕ := 6
def cost_6_apples : ℕ := 21

-- Prove the total cost for 25 apples and the cost for 6 apples at the average price per apple
theorem apple_costs (r1 : ℕ) (r2 : ℕ) (p1 : ℕ) (p2 : ℕ) :
  r1 = 15 → r2 = 25 → p1 = 4 → p2 = 7 → total_apples = 25 → 
  let best_combination_cost := 3 * r2 + r1 in
  best_combination_cost = cost_25_apples ∧
  let avg_cost_per_apple := best_combination_cost / total_apples in
  let cost_for_6_apples := avg_cost_per_apple * 6 in
  cost_for_6_apples = cost_6_apples :=
by
  intros
  sorry

end apple_costs_l384_384293


namespace parametric_to_ordinary_eq_l384_384175

-- Define the parametric equations and the domain of the parameter t
def parametric_eqns (t : ℝ) : ℝ × ℝ := (t + 1, 3 - t^2)

-- Define the target equation to be proved
def target_eqn (x y : ℝ) : Prop := y = -x^2 + 2*x + 2

-- Prove that, given the parametric equations, the target ordinary equation holds
theorem parametric_to_ordinary_eq :
  ∃ (t : ℝ) (x y : ℝ), parametric_eqns t = (x, y) ∧ target_eqn x y :=
by
  sorry

end parametric_to_ordinary_eq_l384_384175


namespace compare_logs_l384_384354

noncomputable def log_base (b x : ℝ) := Real.log x / Real.log b

theorem compare_logs (a b c : ℝ) (h1 : a = log_base 4 1.25) (h2 : b = log_base 5 1.2) (h3 : c = log_base 4 8) :
  c > a ∧ a > b :=
by
  sorry

end compare_logs_l384_384354


namespace functional_equation_solution_l384_384656

theorem functional_equation_solution (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f(f(x) + y) = f(f(x) - y) + 4 * f(x) * y) :
  ∃ C : ℝ, ∀ x : ℝ, f(x) = x^2 + C := 
sorry

end functional_equation_solution_l384_384656


namespace terminating_decimals_count_l384_384734

theorem terminating_decimals_count : 
  let N := 180 in
  let count_terminating_decimals := (List.range' 1 N).filter (λ n, n % 3 = 0) in
  count_terminating_decimals.length = 60 :=
by
  let N := 180
  let count_terminating_decimals := (List.range' 1 N).filter (λ n, n % 3 = 0)
  sorry

end terminating_decimals_count_l384_384734


namespace cheenu_time_difference_l384_384571

-- Define the conditions in terms of Cheenu's activities

variable (boy_run_distance : ℕ) (boy_run_time : ℕ)
variable (midage_bike_distance : ℕ) (midage_bike_time : ℕ)
variable (old_walk_distance : ℕ) (old_walk_time : ℕ)

-- Define the problem with these variables
theorem cheenu_time_difference:
    boy_run_distance = 20 ∧ boy_run_time = 240 ∧
    midage_bike_distance = 30 ∧ midage_bike_time = 120 ∧
    old_walk_distance = 8 ∧ old_walk_time = 240 →
    (old_walk_time / old_walk_distance - midage_bike_time / midage_bike_distance) = 26 := by
    sorry

end cheenu_time_difference_l384_384571


namespace num_divisors_90_l384_384041

theorem num_divisors_90 : (∀ (n : ℕ), n = 90 → (factors n).divisors.card = 12) :=
by {
  intro n,
  intro hn,
  sorry
}

end num_divisors_90_l384_384041


namespace sum_arithmetic_series_base8_l384_384301

theorem sum_arithmetic_series_base8 : 
  let n := 36
  let a := 1
  let l := 30 -- 36_8 in base 10 is 30
  let S := (n * (a + l)) / 2
  let sum_base10 := 558
  let sum_base8 := 1056 -- 558 in base 8 is 1056
  S = sum_base10 ∧ sum_base10 = 1056 :=
by
  sorry

end sum_arithmetic_series_base8_l384_384301


namespace solve_for_x_l384_384140

theorem solve_for_x (x : ℚ) (h : (7 * x) / (x - 2) + 4 / (x - 2) = 6 / (x - 2)) : x = 2 / 7 :=
sorry

end solve_for_x_l384_384140


namespace sufficient_but_not_necessary_condition_l384_384003

noncomputable def f (x a : ℝ) : ℝ := (x + 1) / x + Real.sin x - a^2

theorem sufficient_but_not_necessary_condition (a : ℝ) (h : a = 1) : 
  (∀ x, f x a + f (-x) a = 0) ↔ (a = 1) ∨ (a = -1) :=
by
  sorry

end sufficient_but_not_necessary_condition_l384_384003


namespace sum_of_money_proof_l384_384995

noncomputable def total_sum (A B C : ℝ) : ℝ := A + B + C

theorem sum_of_money_proof (A B C : ℝ) (h1 : B = 0.65 * A) (h2 : C = 0.40 * A) (h3 : C = 64) : total_sum A B C = 328 :=
by 
  sorry

end sum_of_money_proof_l384_384995


namespace terminating_decimals_l384_384700

theorem terminating_decimals (n : ℤ) (h : 1 ≤ n ∧ n ≤ 180) :
  (∃ k : ℤ, k = 20 ∧ ∀ n, 1 ≤ n ∧ n ≤ 180 → (∃ m, m * 180 = n * (2^2 * 5))) := by
  sorry

end terminating_decimals_l384_384700


namespace number_of_students_l384_384623

theorem number_of_students (K M KM : ℕ) (hK : K = 38) (hM : M = 39) (hKM : KM = 32) : K + M - KM = 45 := 
by
  rw [hK, hM, hKM]
  norm_num
  sorry

end number_of_students_l384_384623


namespace num_divisors_first_seven_l384_384148

theorem num_divisors_first_seven (a b : ℤ) (h : 4 * b = 9 - 3 * a) :
  num_divisors (b^2 + 12 * b + 15) {1, 2, 3, 4, 5, 6, 7} = 3 := sorry

end num_divisors_first_seven_l384_384148


namespace smallest_lcm_l384_384418

theorem smallest_lcm (k l : ℕ) (hk : k ≥ 1000) (hl : l ≥ 1000) (huk : k < 10000) (hul : l < 10000) (hk_pos : 0 < k) (hl_pos : 0 < l) (h_gcd: Nat.gcd k l = 5) :
  Nat.lcm k l = 201000 :=
by
  sorry

end smallest_lcm_l384_384418


namespace second_offset_length_l384_384323

theorem second_offset_length (d : ℝ) (h₁ : ℝ) (h₂ : ℝ) (A : ℝ) (h_d : d = 40) (h_h₁ : h₁ = 11) (h_A : A = 400) :
  h₂ = 9 :=
by
  have eq₁ : A = (1/2) * d * h₁ + (1/2) * d * h₂, from sorry
  obtain ⟨H₂⟩ : h₂ = 9, from sorry
  exact H₂

end second_offset_length_l384_384323


namespace problem1_problem2_l384_384126

-- Statement for Problem 1
theorem problem1 (x y : ℝ) (h : (2 * x^2 + 2 * y^2 + 3) * (2 * x^2 + 2 * y^2 - 3) = 27) :
  x^2 + y^2 = 3 := sorry

-- Statement for Problem 2
theorem problem2 (a b c d : ℕ) (h : a * b * c * d = 120) (ha : a < b < c < d) :
  {a, b, c, d} = {2, 3, 4, 5} := sorry

end problem1_problem2_l384_384126


namespace range_of_a_minus_b_l384_384406

theorem range_of_a_minus_b (a b : ℝ) (ha : 1 < a ∧ a < 4) (hb : -2 < b ∧ b < 4) : 
  -3 < a - b ∧ a - b < 6 :=
by
  sorry

end range_of_a_minus_b_l384_384406


namespace second_train_speed_l384_384979

noncomputable def speed_of_second_train (length1 length2 speed1 clearance_time : ℝ) : ℝ :=
  let total_distance := (length1 + length2) / 1000 -- convert meters to kilometers
  let time_in_hours := clearance_time / 3600 -- convert seconds to hours
  let relative_speed := total_distance / time_in_hours
  relative_speed - speed1

theorem second_train_speed : 
  speed_of_second_train 60 280 42 16.998640108791296 = 30.05 := 
by
  sorry

end second_train_speed_l384_384979


namespace arithmetic_sequence_sum_13_l384_384443

variable {a : ℕ → ℤ}

def in_arithmetic_sequence (a : ℕ → ℤ) := ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0

theorem arithmetic_sequence_sum_13 
  (h1 : in_arithmetic_sequence a)
  (h2 : a 2 + a 6 - a 9 = 8)
  (h3 : a 10 - a 3 = 4) :
  (∑ i in finset.range 13, a i) = 156 :=
sorry

end arithmetic_sequence_sum_13_l384_384443


namespace digit_1_more_frequent_than_digit_2_l384_384663

def count_digit_appearance_in_final_sum (lower upper : ℕ) (d : ℕ) : ℕ :=
  (List.range (upper - lower + 1)).countp (λ n => Nat.digits 10 (lower + n)).sumDigits = d

theorem digit_1_more_frequent_than_digit_2 :
  count_digit_appearance_in_final_sum 1 1000000000 1 > count_digit_appearance_in_final_sum 1 1000000000 2 :=
sorry

end digit_1_more_frequent_than_digit_2_l384_384663


namespace mike_picked_12_pears_l384_384088

theorem mike_picked_12_pears (k_picked k_gave_away k_m_together k_left m_left : ℕ) 
  (hkp : k_picked = 47) 
  (hkg : k_gave_away = 46) 
  (hkt : k_m_together = 13)
  (hkl : k_left = k_picked - k_gave_away) 
  (hlt : k_m_left = k_left + m_left) : 
  m_left = 12 := by
  sorry

end mike_picked_12_pears_l384_384088


namespace probability_is_five_sevenths_l384_384916

noncomputable def probability_prime_or_odd : ℚ :=
  let balls := {1, 2, 3, 4, 5, 6, 7}
  let primes := {2, 3, 5, 7}
  let odds := {1, 3, 5, 7}
  let combined := primes ∪ odds
  let favorable_outcomes := combined.card
  let total_outcomes := balls.card
  favorable_outcomes / total_outcomes

theorem probability_is_five_sevenths :
  probability_prime_or_odd = 5 / 7 := 
sorry

end probability_is_five_sevenths_l384_384916


namespace divide_weights_into_equal_groups_l384_384356

theorem divide_weights_into_equal_groups : 
  ∃ (A B C : list ℕ), 
    (A ∪ B ∪ C = {i^2 | i ∈ (finset.range 1 28)}.to_list) ∧
    (A.sum = 2310) ∧ 
    (B.sum = 2310) ∧ 
    (C.sum = 2310) := 
sorry

end divide_weights_into_equal_groups_l384_384356


namespace terminating_decimal_count_number_of_terminating_decimals_l384_384670

theorem terminating_decimal_count {n : ℕ} (h1 : 1 ≤ n) (h2 : n ≤ 180) :
  (∃ k, n = 9 * k ∧ 1 ≤ 9 * k ∧ 9 * k ≤ 180) ↔ n ∈ (finset.range 21).image (λ k, 9 * k) :=
by sorry

theorem number_of_terminating_decimals :
  (finset.range 21).card = 20 :=
by sorry

end terminating_decimal_count_number_of_terminating_decimals_l384_384670


namespace functional_relationship_inversely_proportional_l384_384592

-- Definitions based on conditions
def table_data : List (ℝ × ℝ) := [(100, 1.00), (200, 0.50), (400, 0.25), (500, 0.20)]

-- The main conjecture to be proved
theorem functional_relationship_inversely_proportional (y x : ℝ) (h : (x, y) ∈ table_data) : y = 100 / x :=
sorry

end functional_relationship_inversely_proportional_l384_384592


namespace terminating_decimals_count_l384_384716

theorem terminating_decimals_count :
  ∃ (count : ℕ), count = 60 ∧
  ∀ (n : ℕ), 1 ≤ n ∧ n ≤ 180 →
  (∃ m : ℕ, m * 180 = n * (2^2 * 5)) → 60 = count := sorry

end terminating_decimals_count_l384_384716


namespace initial_teach_count_l384_384290

theorem initial_teach_count :
  ∃ (x y : ℕ), (x + x * y + (x + x * y) * (y + x * y) = 195) ∧
               (y + x * y + (y + x * y) * (x + x * y) = 192) ∧
               x = 5 ∧ y = 2 :=
by {
  sorry
}

end initial_teach_count_l384_384290


namespace volume_of_combined_solid_l384_384262

theorem volume_of_combined_solid :
  let r := 8
  let h_cylinder := 16
  let height_wedge := h_cylinder
  let radius_cone := r
  let height_cone := h_cylinder
  let V_cylinder := (r * r * h_cylinder * Real.pi) / 2
  let V_cone := (1 / 3) * Real.pi * (radius_cone * radius_cone) * height_cone
  let total_volume := V_cylinder + V_cone
  total_volume = (2560 / 3) * Real.pi :=
by
  let r := 8
  let h_cylinder := 16
  let height_wedge := h_cylinder
  let radius_cone := r
  let height_cone := h_cylinder
  let V_cylinder := (r * r * h_cylinder * Real.pi) / 2
  let V_cone := (1 / 3) * Real.pi * (radius_cone * radius_cone) * height_cone
  let total_volume := V_cylinder + V_cone
  calc
    total_volume = 512 * Real.pi + 1024 / 3 * Real.pi : by sorry
               ... = 2560 / 3 * Real.pi : by sorry

end volume_of_combined_solid_l384_384262


namespace max_value_of_cubes_l384_384473

theorem max_value_of_cubes 
  (x y z : ℝ) 
  (h : x^2 + y^2 + z^2 = 9) : 
  x^3 + y^3 + z^3 ≤ 27 :=
  sorry

end max_value_of_cubes_l384_384473


namespace sin_double_angle_neg_one_l384_384816

theorem sin_double_angle_neg_one (α : ℝ) (a b : ℝ × ℝ) (h₁ : a = (1, Real.cos α)) (h₂ : b = (Real.sin α, 1)) (h₃ : a.1 * b.1 + a.2 * b.2 = 0) :
  Real.sin (2 * α) = -1 :=
sorry

end sin_double_angle_neg_one_l384_384816


namespace projection_of_AB_on_DA_dot_product_AD_BE_l384_384064

noncomputable def equilateral_triangle (A B C: Point) : Prop :=
  dist A B = 1 ∧ dist B C = 1 ∧ dist C A = 1

noncomputable def point_D_condition (A B C D: Point) : Prop :=
  let BC := vector B C in
  let BD := vector B D in
  BC = 2 • BD

noncomputable def point_E_condition (A C E: Point) : Prop :=
  let CA := vector C A in
  let CE := vector C E in
  CA = 3 • CE

theorem projection_of_AB_on_DA 
  {A B C D: Point} (h_tri: equilateral_triangle A B C)
  (hD: point_D_condition A B C D) :
  let DA := vector D A
  let AB := vector A B
  projection AB DA = - (real.sqrt 3) / 2 :=
sorry

theorem dot_product_AD_BE 
  {A B C D E: Point} (h_tri: equilateral_triangle A B C)
  (hD: point_D_condition A B C D)
  (hE: point_E_condition A C E):
  let AD := vector A D
  let BE := vector B E
  AD • BE = - 1 / 4 :=
sorry

end projection_of_AB_on_DA_dot_product_AD_BE_l384_384064


namespace chord_length_sqrt_10_l384_384370

/-
  Given a line L: 3x - y - 6 = 0 and a circle C: x^2 + y^2 - 2x - 4y = 0,
  prove that the length of the chord AB formed by their intersection is sqrt(10).
-/

noncomputable def line_L : Set (ℝ × ℝ) := {p | ∃ x y, p = (x, y) ∧ 3 * x - y - 6 = 0}

noncomputable def circle_C : Set (ℝ × ℝ) := {p | ∃ x y, p = (x, y) ∧ x^2 + y^2 - 2 * x - 4 * y = 0}

noncomputable def chord_length (L C : Set (ℝ × ℝ)) : ℝ :=
  let center := (1, 2)
  let r := Real.sqrt 5
  let d := |3 * 1 - 2 - 6| / Real.sqrt (1 + 3^2)
  2 * Real.sqrt (r^2 - d^2)

theorem chord_length_sqrt_10 : chord_length line_L circle_C = Real.sqrt 10 := sorry

end chord_length_sqrt_10_l384_384370


namespace radius_of_triangle_DEF_l384_384220

noncomputable def radius_of_inscribed_circle (DE DF EF : ℝ) : ℝ :=
  let s := (DE + DF + EF) / 2
  let K := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF))
  K / s

theorem radius_of_triangle_DEF :
  radius_of_inscribed_circle 26 15 17 = 121 / 29 := by
sorry

end radius_of_triangle_DEF_l384_384220


namespace geometric_sequence_a4_l384_384162

theorem geometric_sequence_a4 (x a_4 : ℝ) (h1 : 2*x + 2 = (3*x + 3) * (2*x + 2) / x)
  (h2 : x = -4 ∨ x = -1) (h3 : x = -4) : a_4 = -27 / 2 :=
by
  sorry

end geometric_sequence_a4_l384_384162


namespace least_y_l384_384216

theorem least_y (y : ℝ) : (2 * y ^ 2 + 7 * y + 3 = 5) → y = -2 :=
sorry

end least_y_l384_384216


namespace solve_equation_l384_384921

theorem solve_equation (x : ℝ) :
  (1 / (x + 10) + 1 / (x + 8) = 1 / (x + 11) + 1 / (x + 7)) → x = -9 :=
by 
  sorry

end solve_equation_l384_384921


namespace dutch_americans_blue_shirts_window_seats_l384_384112

theorem dutch_americans_blue_shirts_window_seats :
  let total_people := 90
  let dutch_fraction := 3 / 5
  let dutch_americans_fraction := 1 / 2
  let window_seats_fraction := 1 / 3
  let blue_shirts_fraction := 2 / 3
  total_people * dutch_fraction * dutch_americans_fraction * window_seats_fraction * blue_shirts_fraction = 6 := by
  sorry

end dutch_americans_blue_shirts_window_seats_l384_384112


namespace lines_intersect_lines_perpendicular_l384_384815

variables {A₁ A₂ B₁ B₂ C₁ C₂ : ℝ}

-- Line equations
def l1 (x y : ℝ) : Prop := A₁ * x + B₁ * y + C₁ = 0
def l2 (x y : ℝ) : Prop := A₂ * x + B₂ * y + C₂ = 0

-- Condition statements
def C₁_neq_C₂ : Prop := C₁ ≠ C₂
def AB_nonzero : Prop := A₁ * B₂ - A₂ * B₁ ≠ 0
def AB_zero : Prop := A₁ * B₂ - A₂ * B₁ = 0
def perpendicular : Prop := A₁ * A₂ + B₁ * B₂ = 0

-- Theorems
theorem lines_intersect (h : AB_nonzero) : ∃ x y : ℝ, l1 x y ∧ l2 x y := by sorry

theorem lines_perpendicular (h : perpendicular) : 
  (∃ (x₁ y₁ x₂ y₂ : ℝ), (l1 x₁ y₁ ∧ l2 x₂ y₂) ∧ 
  (A₁ = 0 ∨ B₁ = 0 ∨ A₂ = 0 ∨ B₂ = 0 ∨ 
  (A₁ * B₂ ≠ 0 ∧ A₂ * B₁ ≠ 0 ∧ (A₁ / B₁) * (A₂ / B₂) = -1))) := by sorry

end lines_intersect_lines_perpendicular_l384_384815


namespace range_of_a_l384_384395

noncomputable def f (a x : ℝ) : ℝ := a * x^2 + x

theorem range_of_a {a : ℝ} :
  (∀ x ∈ Ioc 0 1, |f a x| ≤ 1) ↔ -2 ≤ a ∧ a < 0 :=
by
  unfold f
  sorry

end range_of_a_l384_384395


namespace sum_of_digits_of_smallest_palindromic_prime_gt_1000_l384_384338

def is_palindromic (n : ℕ) : Prop :=
  let s := n.repr in s = s.reverse

def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits_of_smallest_palindromic_prime_gt_1000 :
  ∃ p : ℕ, p > 1000 ∧ is_palindromic p ∧ Prime p ∧ digit_sum p = 18 :=
by 
  use 1881
  split; try {simp}
  sorry -- Proof steps demonstrating the smallest palindromic prime greater than 1000 and its digit sum being 18

end sum_of_digits_of_smallest_palindromic_prime_gt_1000_l384_384338


namespace hare_overtakes_tortoise_l384_384431

noncomputable def hare_distance (t: ℕ) : ℕ := 
  if t ≤ 5 then 10 * t
  else if t ≤ 20 then 50
  else 50 + 20 * (t - 20)

noncomputable def tortoise_distance (t: ℕ) : ℕ :=
  2 * t

theorem hare_overtakes_tortoise : 
  ∃ t : ℕ, t ≤ 60 ∧ hare_distance t = tortoise_distance t ∧ 60 - t = 22 :=
sorry

end hare_overtakes_tortoise_l384_384431


namespace sin_cos_identity_l384_384574

theorem sin_cos_identity (α : ℝ) : 
  sin (α / 2) ^ 6 - cos (α / 2) ^ 6 = ((sin α ^ 2 - 4) / 4) * cos α :=
by
  sorry

end sin_cos_identity_l384_384574


namespace find_x_for_f_eq_10_l384_384165

noncomputable def f : ℝ → ℝ :=
  λ x, if x ≤ 0 then x^2 + 1 else -2 * x

theorem find_x_for_f_eq_10 (x : ℝ) (h : f x = 10) : x = -3 :=
sorry

end find_x_for_f_eq_10_l384_384165


namespace sum_of_angles_l384_384178

theorem sum_of_angles (θ : Fin 6 → ℝ) (hθ : ∀ k : Fin 6, 0 ≤ θ k ∧ θ k < 360) 
  (h_roots : ∃ (roots : Fin 6 → ℂ), ∀ k : Fin 6, roots k = complex.cis (θ k) ∧ (roots k) ^ 6 = -1 + complex.I) :
  ∑ i, θ i = 1125 :=
sorry

end sum_of_angles_l384_384178


namespace terminating_decimals_l384_384697

theorem terminating_decimals (n : ℤ) (h : 1 ≤ n ∧ n ≤ 180) :
  (∃ k : ℤ, k = 20 ∧ ∀ n, 1 ≤ n ∧ n ≤ 180 → (∃ m, m * 180 = n * (2^2 * 5))) := by
  sorry

end terminating_decimals_l384_384697


namespace determine_unit_prices_most_cost_effective_store_l384_384283

-- Definitions of variables and conditions
def unit_price_shoes (x : ℕ) := x
def unit_price_sportswear (y : ℕ) := y

axiom condition1 (x y : ℕ) : x + y = 516
axiom condition2 (x y : ℕ) : y = 3 * x - 12

-- First question: determining the unit prices
theorem determine_unit_prices (x y : ℕ) (h1 : condition1 x y) (h2 : condition2 x y) :
  x = 132 ∧ y = 384 :=
sorry

-- Definitions for store discounts
def store_a_discount (total : ℕ) : ℕ := total * 80 / 100
def store_b_discount (total : ℕ) : ℕ := total - 25 * (total / 100)

-- Second question: determining the most cost-effective store
theorem most_cost_effective_store (total : ℕ) (h1 : total = 516) :
  store_b_discount total < store_a_discount total :=
sorry

end determine_unit_prices_most_cost_effective_store_l384_384283


namespace total_artworks_created_l384_384891

theorem total_artworks_created
  (students_group1 : ℕ := 24) (students_group2 : ℕ := 12)
  (kits_total : ℕ := 48)
  (kits_per_3_students : ℕ := 3) (kits_per_2_students : ℕ := 2)
  (artwork_types : ℕ := 3)
  (paintings_group1_1 : ℕ := 12 * 2) (drawings_group1_1 : ℕ := 12 * 4) (sculptures_group1_1 : ℕ := 12 * 1)
  (paintings_group1_2 : ℕ := 12 * 1) (drawings_group1_2 : ℕ := 12 * 5) (sculptures_group1_2 : ℕ := 12 * 3)
  (paintings_group2_1 : ℕ := 4 * 3) (drawings_group2_1 : ℕ := 4 * 6) (sculptures_group2_1 : ℕ := 4 * 3)
  (paintings_group2_2 : ℕ := 8 * 4) (drawings_group2_2 : ℕ := 8 * 7) (sculptures_group2_2 : ℕ := 8 * 1)
  : (paintings_group1_1 + paintings_group1_2 + paintings_group2_1 + paintings_group2_2) +
    (drawings_group1_1 + drawings_group1_2 + drawings_group2_1 + drawings_group2_2) +
    (sculptures_group1_1 + sculptures_group1_2 + sculptures_group2_1 + sculptures_group2_2) = 336 :=
by sorry

end total_artworks_created_l384_384891


namespace factor_expression_l384_384302

theorem factor_expression (x : ℝ) : 9 * x^2 + 3 * x = 3 * x * (3 * x + 1) := 
by
  sorry

end factor_expression_l384_384302


namespace remainder_of_70th_number_in_set_s_l384_384482

theorem remainder_of_70th_number_in_set_s (s : Set ℕ) (h₁ : ∀ n ∈ s, n % 8 = r)
  (h₂ : ∃ k, k = 70 ∧ nth s k = 557) : r = 5 := 
by 
  sorry

end remainder_of_70th_number_in_set_s_l384_384482


namespace lhopital_rule_l384_384480

open Filter

variables {α β : Type*} [linear_order α] [topological_space α] [topological_space β] 
           [order_topology β] [normed_field β] {f g : α → β} {a A : β}

theorem lhopital_rule (hfc : ∀ {x y : α}, x ≠ y → ∃ c ∈ Ioo (min x y) (max x y), (g y - g x) * (f x - f y) = (f' c * (g y - g x))) 
                      (hf : f a = 0) (hg : g a = 0) 
                      (lim_hyp : tendsto (λ x, (deriv f x / deriv g x)) (nhds_within a (Ioi a)) (nhds A)) :
  tendsto (λ x, f x / g x) (nhds_within a (Ioi a)) (nhds A) := by
  sorry

end lhopital_rule_l384_384480


namespace tina_wins_more_than_losses_l384_384971

theorem tina_wins_more_than_losses 
  (initial_wins : ℕ)
  (additional_wins : ℕ)
  (first_loss : ℕ)
  (doubled_wins : ℕ)
  (second_loss : ℕ)
  (total_wins : ℕ)
  (total_losses : ℕ)
  (final_difference : ℕ) :
  initial_wins = 10 →
  additional_wins = 5 →
  first_loss = 1 →
  doubled_wins = 30 →
  second_loss = 1 →
  total_wins = initial_wins + additional_wins + doubled_wins →
  total_losses = first_loss + second_loss →
  final_difference = total_wins - total_losses →
  final_difference = 43 :=
by
  sorry

end tina_wins_more_than_losses_l384_384971


namespace round_robin_tournament_l384_384437

theorem round_robin_tournament :
  ∃ (n : ℕ), n = 21 ∧
  (∀ (sets_of_three : ℕ),
   sets_of_three = (nat.choose 21 3) - 21 * (nat.choose 11 2) ∧ 
   sets_of_three = 175) :=
begin
  -- assumption: there are 21 teams
  let n := 21,
  use n,
  split,
  { exact rfl },
  {
    use nat.choose 21 3 - 21 * nat.choose 11 2,
    split,
    { 
      -- calculate total sets of three
      have h1 : nat.choose 21 3 = (21 * 20 * 19) / (3 * 2 * 1), from rfl,
      -- calculate single direction domination sets
      have h2 : 21 * nat.choose 11 2 = 21 * ((11 * 10) / (2 * 1)), from rfl,
      -- subtract and confirm they match the calculation above
      exact rfl,
    },
    { -- show that the number of sets with directed cycle is 175
      exact rfl,
    }
  }
end

end round_robin_tournament_l384_384437


namespace least_y_l384_384215

theorem least_y (y : ℝ) : (2 * y ^ 2 + 7 * y + 3 = 5) → y = -2 :=
sorry

end least_y_l384_384215


namespace max_knights_in_unfortunate_island_l384_384117

theorem max_knights_in_unfortunate_island (natives : ℕ) (liars_made_a_mistake : ℕ) : 
  natives = 2022 → liars_made_a_mistake = 3 → 
  ∃ (knights : ℕ), knights = 1349 :=
by
  intros h_natives h_mistake
  use 1349
  sorry

end max_knights_in_unfortunate_island_l384_384117


namespace k_20_coloring_connected_l384_384542

open SimpleGraph

theorem k_20_coloring_connected :
  ∃ c : Fin 5, 
  ∀ (K : SimpleGraph (Fin 20)) 
    (colored_K : ∀ e : K.edgeSet, Fin 5),
    (K = completeGraph (Fin 20)) → 
    ¬(K.deleteEdges (colored_K⁻¹' {c})).Disconnected :=
sorry

end k_20_coloring_connected_l384_384542


namespace mineral_samples_per_shelf_l384_384299

theorem mineral_samples_per_shelf (total_samples : ℕ) (num_shelves : ℕ) (h1 : total_samples = 455) (h2 : num_shelves = 7) :
  total_samples / num_shelves = 65 :=
by
  sorry

end mineral_samples_per_shelf_l384_384299


namespace largest_prime_factor_1023_l384_384565

theorem largest_prime_factor_1023 : 
  (∃ p, p ∣ 1023 ∧ nat.prime p) ∧ 
  (∀ q, q ∣ 1023 ∧ nat.prime q → q ≤ 31) :=
by
  let p3 := 3
  let p11 := 11
  let p31 := 31
  have h1 : 1023 = p3 * 341 := rfl
  have h2 : 341 = p11 * p31 := rfl
  have h3 : nat.prime p3 := by sorry
  have h4 : nat.prime p11 := by sorry
  have h5 : nat.prime p31 := by sorry
  -- Proof of the theorem using these conditions
  sorry

end largest_prime_factor_1023_l384_384565


namespace perpendicular_line_through_point_eq_l384_384641

theorem perpendicular_line_through_point_eq {x y : ℝ} :
  (∀ (x y : ℝ), 3 * x - 6 * y = 9) → 
  (2, -3) →
  (∀ (m : ℝ), m = -2 → y = -2 * x + 1) :=
by
  intros h point m_eq
  sorry

end perpendicular_line_through_point_eq_l384_384641


namespace average_birds_monday_l384_384089

variable (M : ℕ)

def avg_birds_monday (M : ℕ) : Prop :=
  let total_sites := 5 + 5 + 10
  let total_birds := 5 * M + 5 * 5 + 10 * 8
  (total_birds = total_sites * 7)

theorem average_birds_monday (M : ℕ) (h : avg_birds_monday M) : M = 7 := by
  sorry

end average_birds_monday_l384_384089


namespace sum_logs_geom_seq_zero_l384_384468

noncomputable def geom_sequence (a1 r : ℝ) (n : ℕ) : ℝ :=
  a1 * r^(n-1)

noncomputable def log_sum (a1 r : ℝ) (n : ℕ) : ℝ :=
  (finset.range n).sum (λ i, real.log (geom_sequence a1 r (i + 1)))

theorem sum_logs_geom_seq_zero (a1 r : ℝ) (m n : ℕ) (hmn : m ≠ n) (h_sum: log_sum a1 r m = log_sum a1 r n) :
    log_sum a1 r (m + n) = 0 :=
begin
  sorry
end

end sum_logs_geom_seq_zero_l384_384468


namespace smallest_percent_increase_between_Q2_and_Q3_l384_384662

def values : List ℕ := [100, 300, 400, 700, 1500, 3000, 6000, 12000, 24000, 48000, 96000, 180000, 360000, 720000, 1440000]

def percent_increase (v1 v2 : ℕ) : ℕ := ((v2 - v1) * 100) / v1

theorem smallest_percent_increase_between_Q2_and_Q3 :
  let increases := values.zip(values.tail) |>.map (fun (v1, v2) => percent_increase v1 v2)
  increases.get? 1 == some 33 := by
{
  let values_zip_tail := values.zip values.tail
  have increases_list : List ℕ := values_zip_tail.map (fun (v1, v2) => percent_increase v1 v2)
  have Q2_to_Q3_increase : percent_increase (values.get? 1).get! (values.get? 2).get! = 33 := by
  {
    unfold percent_increase,
    calc
      ((400 - 300) * 100) / 300 = (100 * 100) / 300 : by ring  -- Calculated manually
      ... = 33.33              : by norm_cast
  }
  have smallest_index : Nat := 1
  have smallest_value : ℕ := 33
  calc
    (increases.get? 1).get ~= smallest_value : by sorry    -- Skip proof
}

end smallest_percent_increase_between_Q2_and_Q3_l384_384662


namespace final_number_not_zero_l384_384114

/-- Define the sum of the first n natural numbers -/
def sum_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Main theorem: On a board with numbers from 1 to 1985, if we repeatedly replace any two numbers with the absolute value of their difference, the final remaining number cannot be zero. -/
theorem final_number_not_zero :
  ¬ ∃ (final_number : ℕ), final_number = 0 ∧
  ∀ (numbers : list ℕ), (∀ (n : ℕ), n ∈ numbers → n ≤ 1985) →
  (∀ (n : ℕ), n > 0) →
  (∀ (a b : ℕ), a ∈ numbers → b ∈ numbers →
    (numbers.erase a).erase b ++ [|a - b|] = numbers.erase a ∩ (numbers.erase b).erase b ++ [|a - b|]) →
  (numbers.length = 1 → final_number ∈ numbers)
:= sorry

end final_number_not_zero_l384_384114


namespace number_of_divisors_l384_384039

-- Defining the given number and its prime factorization as a condition.
def given_number : ℕ := 90

-- Defining the prime factorization.
def prime_factorization (n : ℕ) : List (ℕ × ℕ) :=
  if n = 90 then [(2, 1), (3, 2), (5, 1)] else []

-- The statement to prove that the number of positive divisors of 90 is 12.
theorem number_of_divisors (n : ℕ) (pf : List (ℕ × ℕ)) :
  n = 90 → pf = [(2, 1), (3, 2), (5, 1)] →
  (pf.map (λ p, p.2 + 1)).prod = 12 :=
by
  intros hn hpf
  rw [hn, hpf]
  simp
  sorry

end number_of_divisors_l384_384039


namespace range_of_c_l384_384004

-- Definitions of the propositions p and q
def p (c : ℝ) : Prop := ∀ x y : ℝ, x < y → c^x > c^y
def q (c : ℝ) : Prop := ∃ x : ℝ, x^2 - c^2 ≤ - (1 / 16)

-- Main theorem
theorem range_of_c (c : ℝ) (h1 : c > 0) (h2 : p c) (h3 : q c) : c ≥ 1 / 4 ∧ c < 1 :=
  sorry

end range_of_c_l384_384004


namespace number_of_sets_B_l384_384833

def A : Set ℤ := {x | 3 * x - x^2 > 0}

theorem number_of_sets_B :
  ∃ (B : Set (Set ℤ)), B.card = 4 ∧ ∀ b ∈ B, A ∪ b = {1, 2, 3, 4} :=
by
  sorry

end number_of_sets_B_l384_384833


namespace increasing_function_and_f1_eq_2_l384_384166

variables {ℝ : Type*} [LinearOrderedField ℝ]

-- Definitions

def f (x : ℝ) : ℝ

axiom functional_eqn : ∀ x y : ℝ, f(x + y) = f(x) + f(y) - 1
axiom pos_x_implies_f_gt_1 : ∀ x : ℝ, x > 0 → f(x) > 1
axiom f_3_eq_4 : f(3) = 4

-- Theorem to prove the function is increasing on ℝ and f(1) = 2
theorem increasing_function_and_f1_eq_2 : (∀ x1 x2 : ℝ, x1 > x2 → f(x1) > f(x2)) ∧ (f(1) = 2) :=
by
  sorry

end increasing_function_and_f1_eq_2_l384_384166


namespace terminating_decimals_l384_384714

theorem terminating_decimals (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 180) :
  (∃ k : ℕ, k = 20 ∧ ∀ m : ℕ, 1 ≤ m ∧ m ≤ 180 → (decimal_terminates (m / 180) ↔ m % 9 = 0)) :=
by {
  sorry
}

def decimal_terminates (x : ℚ) : Prop :=
  sorry

end terminating_decimals_l384_384714


namespace find_x_l384_384785

theorem find_x {x y : ℝ} (hx : x ≠ 0) (hy : y ≠ 0)
    (h1 : x + 1/y = 10) (h2 : y + 1/x = 5/12) : x = 4 ∨ x = 6 :=
by
  sorry

end find_x_l384_384785


namespace find_m_l384_384428

theorem find_m (m : ℝ) (h : |m - 4| = |2 * m + 7|) : m = -11 ∨ m = -1 :=
sorry

end find_m_l384_384428


namespace son_age_is_14_l384_384271

variable (son father : ℕ)

noncomputable def present_age_of_son (son : ℕ) :=
  ∃ father : ℕ, father = son + 40 ∧ father + 6 = 3 * (son + 6)

theorem son_age_is_14 : present_age_of_son 14 :=
by {
  use 54,
  split,
  {
    -- first condition: father = son + 40
    exact rfl,
  },
  {
    -- second condition: father + 6 = 3 * (son + 6)
    exact rfl,
  }
}

end son_age_is_14_l384_384271


namespace video_game_cost_l384_384151

def tanner_savings_september : ℕ := 17
def tanner_savings_october : ℕ := 48
def tanner_savings_november : ℕ := 25
def amount_left_after_purchase : ℕ := 41

def total_savings : ℕ :=
  tanner_savings_september + tanner_savings_october + tanner_savings_november

def cost_of_video_game : ℕ :=
  total_savings - amount_left_after_purchase

theorem video_game_cost :
  cost_of_video_game = 49 :=
by
  rw [cost_of_video_game, total_savings]
  rw [Nat.add_assoc, Nat.add_comm 25, Nat.add_assoc]
  simp [tanner_savings_september, tanner_savings_october, tanner_savings_november, amount_left_after_purchase]
  sorry

end video_game_cost_l384_384151


namespace factor_t_squared_minus_81_l384_384320

theorem factor_t_squared_minus_81 (t : ℝ) : t^2 - 81 = (t - 9) * (t + 9) :=
by
  sorry

end factor_t_squared_minus_81_l384_384320


namespace girls_left_class_l384_384527

variable (G B G₂ B₁ : Nat)

theorem girls_left_class (h₁ : 5 * B = 6 * G) 
                         (h₂ : B = 120)
                         (h₃ : 2 * B₁ = 3 * G₂)
                         (h₄ : B₁ = B) : 
                         G - G₂ = 20 :=
by
  sorry

end girls_left_class_l384_384527


namespace maximize_length_of_sum_l384_384760

noncomputable def maximise_vector_sum (vectors : List ℝ × List ℝ) : List ℝ × List ℝ :=
  sorry -- Function that finds the vectors summing to the maximum length

theorem maximize_length_of_sum :
  ∀ (vectors : List ℝ × List ℝ) (n : ℕ), 
    length vectors = 25 ∧ 
    regular_pentagon vectors ∧
    n = 12 → 
    ∃ subset : List (ℝ × ℝ), 
      length subset = n ∧ 
      (sum_vectors subset = maximise_vector_sum vectors) :=
  by
  intro vectors n
  intro H
  sorry

end maximize_length_of_sum_l384_384760


namespace solution_l384_384452

def num_warehouse_workers := 4
def num_managers := 2
def wage_warehouse_worker_hour := 15
def wage_manager_hour := 20
def FICA_rate := 0.10
def work_hours_per_day := 8
def total_monthly_expense := 22000

def daily_wage_warehouse_workers := num_warehouse_workers * wage_warehouse_worker_hour * work_hours_per_day
def daily_wage_managers := num_managers * wage_manager_hour * work_hours_per_day
def daily_wage_total := daily_wage_warehouse_workers + daily_wage_managers

def monthly_wage_without_taxes := total_monthly_expense / (1 + FICA_rate)

def days_worked_per_month := monthly_wage_without_taxes / daily_wage_total

theorem solution :
  days_worked_per_month = 25 := by
  sorry

end solution_l384_384452


namespace parity_of_functions_l384_384908

def f (x : ℝ) : ℝ := 3 * Real.sin x
def g (x : ℝ) : ℝ := 3 + Real.cos x

theorem parity_of_functions :
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x : ℝ, g (-x) = g x) :=
by {
  sorry -- skips the actual proof implementation
}

end parity_of_functions_l384_384908


namespace root_in_interval_l384_384342

def polynomial (x : ℝ) := x^3 + 3 * x^2 - x + 1

noncomputable def A : ℤ := -4
noncomputable def B : ℤ := -3

theorem root_in_interval : (∃ x : ℝ, polynomial x = 0 ∧ (A : ℝ) < x ∧ x < (B : ℝ)) :=
sorry

end root_in_interval_l384_384342


namespace sequence_general_term_l384_384516

def sequence (n : ℕ) : ℕ :=
  if n = 1 then 3 else
  if n = 2 then 5 else
  if n = 3 then 9 else
  if n = 4 then 17 else
  if n = 5 then 33 else
  sorry -- Placeholder for other terms

theorem sequence_general_term (n : ℕ) : sequence n = 2^n + 1 :=
by sorry

end sequence_general_term_l384_384516


namespace train_passes_man_in_approx_18_seconds_l384_384576

noncomputable def train_length : ℝ := 300 -- meters
noncomputable def train_speed : ℝ := 68 -- km/h
noncomputable def man_speed : ℝ := 8 -- km/h
noncomputable def kmh_to_mps (v : ℝ) : ℝ := v * 1000 / 3600
noncomputable def relative_speed_mps : ℝ := kmh_to_mps (train_speed - man_speed)
noncomputable def time_to_pass_man : ℝ := train_length / relative_speed_mps

theorem train_passes_man_in_approx_18_seconds :
  abs (time_to_pass_man - 18) < 1 :=
by
  sorry

end train_passes_man_in_approx_18_seconds_l384_384576


namespace incenter_relation_l384_384458

variable (A B C D : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D]
variable {rB rC r b c : ℝ}
variable {angle_CAB : ℝ}
variable {BC AC AB : ℝ}

-- Assume we have a triangle, certain angles, and certain lengths.
-- These are our hypotheses:
axiom h1 : angle_CAB = 60
axiom h2 : ∃ D, IsAngleBisector A D BC
axiom h3 : IncircleRadius B D A = rB
axiom h4 : IncircleRadius C D A = rC
axiom h5 : IncircleRadius A B C = r
axiom h6 : SideLength A C = b
axiom h7 : SideLength A B = c

-- Our goal to prove:
theorem incenter_relation :
  (1 / rB) + (1 / rC) = 2 * (1 / r + 1 / b + 1 / c) :=
sorry

end incenter_relation_l384_384458


namespace fruits_given_away_l384_384863

-- Definitions based on the conditions
def initial_pears := 10
def initial_oranges := 20
def initial_apples := 2 * initial_pears
def initial_fruits := initial_pears + initial_oranges + initial_apples
def fruits_left := 44

-- Theorem to prove the total number of fruits given to her sister
theorem fruits_given_away : initial_fruits - fruits_left = 6 := by
  sorry

end fruits_given_away_l384_384863


namespace terminating_decimals_l384_384704

theorem terminating_decimals (n : ℤ) (h : 1 ≤ n ∧ n ≤ 180) :
  (∃ k : ℤ, k = 20 ∧ ∀ n, 1 ≤ n ∧ n ≤ 180 → (∃ m, m * 180 = n * (2^2 * 5))) := by
  sorry

end terminating_decimals_l384_384704


namespace carlos_goal_l384_384636

def july_books : ℕ := 28
def august_books : ℕ := 30
def june_books : ℕ := 42

theorem carlos_goal (goal : ℕ) :
  goal = june_books + july_books + august_books := by
  sorry

end carlos_goal_l384_384636


namespace red_ball_higher_probability_l384_384547

-- Define the problem parameters and conditions
def ball_probability (k : ℕ) : ℚ := 3^(-k)

-- Non-constructive definition to simplify probability results handling
noncomputable def total_probability_same_bin : ℚ := (∑' k : ℕ, 3^(-3 * (k + 1))) -- summation from k=1 to infinity

-- Auxiliary definition to extract the resultant probability
noncomputable def red_higher_probability : ℚ := (1 - total_probability_same_bin) / 3

-- Define the main theorem capturing the problem
theorem red_ball_higher_probability :
  red_higher_probability = 25 / 78 :=
by
  sorry

end red_ball_higher_probability_l384_384547


namespace solve_fraction_equation_l384_384923

theorem solve_fraction_equation (x : ℝ) :
  (1 / (x + 10) + 1 / (x + 8) = 1 / (x + 11) + 1 / (x + 7)) ↔ x = -9 :=
by {
  sorry
}

end solve_fraction_equation_l384_384923


namespace sum_of_two_lowest_scores_l384_384454

theorem sum_of_two_lowest_scores {A B C D : ℝ} (h : [A, B, C, D].sorted (≤)) 
  (h_avg_before : (A + B + C + D) / 4 = 35) 
  (h_avg_after : (C + D) / 2 = 40) : 
  A + B = 60 := 
sorry

end sum_of_two_lowest_scores_l384_384454


namespace john_hourly_wage_with_bonus_l384_384866

structure JohnJob where
  daily_wage : ℕ
  work_hours : ℕ
  bonus_amount : ℕ
  extra_hours : ℕ

def total_daily_wage (job : JohnJob) : ℕ :=
  job.daily_wage + job.bonus_amount

def total_work_hours (job : JohnJob) : ℕ :=
  job.work_hours + job.extra_hours

def hourly_wage (job : JohnJob) : ℕ :=
  total_daily_wage job / total_work_hours job

noncomputable def johns_job : JohnJob :=
  { daily_wage := 80, work_hours := 8, bonus_amount := 20, extra_hours := 2 }

theorem john_hourly_wage_with_bonus :
  hourly_wage johns_job = 10 :=
by
  sorry

end john_hourly_wage_with_bonus_l384_384866


namespace remainder_of_q_div_x_plus_2_l384_384095

noncomputable def q (x : ℝ) : ℝ := 2 * x^6 - 3 * x^4 + D * x^2 + 6

theorem remainder_of_q_div_x_plus_2 (D : ℝ) (h : q 2 = 14) : q (-2) = 158 :=
by
  unfold q
  sorry

end remainder_of_q_div_x_plus_2_l384_384095


namespace probability_point_in_circle_l384_384960

theorem probability_point_in_circle (radius : ℝ) (side_length : ℝ) : 
  radius = 2 → side_length = 4 → 
  let area_square := side_length ^ 2 in
  let area_circle := π * radius ^ 2 in
  (area_circle / area_square) = π / 4 :=
by
  intro h_radius h_side_length
  let area_square := side_length ^ 2
  let area_circle := π * radius ^ 2
  have : area_circle / area_square = π / 4 := sorry
  exact this

end probability_point_in_circle_l384_384960


namespace like_terms_correct_answer_l384_384226

-- Definitions of the conditions given in the problem
def optionA_expr1 := -6 * x * y
def optionA_expr2 := x * z

def optionB_expr1 := 4 * x^2 * y
def optionB_expr2 := (0.5:ℝ) * x * y^2

def optionC_expr1 := (1/3) * x^2 * y
def optionC_expr2 := -y * x^2

def optionD_expr1 := 2 * x * y
def optionD_expr2 := 3 * x * y * z

-- Definition to check if two expressions are like terms
def are_like_terms (expr1 expr2 : ℝ) : Prop :=
  ∃ (c1 c2 : ℝ) (v1 v2 : List (ℝ × ℝ)), expr1 = c1 * v1.prod ∧ expr2 = c2 * v2.prod ∧ (v1 = v2)

-- The theorem to prove the correct answer
theorem like_terms_correct_answer : are_like_terms optionC_expr1 optionC_expr2 :=
  sorry

end like_terms_correct_answer_l384_384226


namespace terminating_decimal_count_l384_384748

theorem terminating_decimal_count : 
  (Finset.filter (λ n, n % 3 = 0) (Finset.range 181)).card = 60 := by
  sorry

end terminating_decimal_count_l384_384748


namespace part1_solution_set_part2_range_of_m_l384_384798

def f (x : ℝ) := 1 - (abs (x - 2))

theorem part1_solution_set :
  {x : ℝ | f x > 1 - abs (x + 4)} = set.Ioi (-1) :=
by sorry

theorem part2_range_of_m (m : ℝ) :
  (∀ x ∈ set.Ioo 2 (5 / 2), f x > abs (x - m)) ↔ (2 ≤ m ∧ m < 3) :=
by sorry

end part1_solution_set_part2_range_of_m_l384_384798


namespace simplify_expression_l384_384138

noncomputable def x : ℝ := Real.sqrt 3 + 1

theorem simplify_expression (x : ℝ) (h : x = Real.sqrt 3 + 1) :
  ((1 - (x / (x + 1))) / ((x^2 - 1) / (x^2 + 2*x + 1))) = Real.sqrt 3 / 3 :=
by
  sorry

end simplify_expression_l384_384138


namespace divisors_of_90_l384_384035

def num_pos_divisors (n : ℕ) : ℕ :=
  let factors := if n = 90 then [(2, 1), (3, 2), (5, 1)] else []
  factors.foldl (fun acc (p, k) => acc * (k + 1)) 1

theorem divisors_of_90 : num_pos_divisors 90 = 12 := by
  sorry

end divisors_of_90_l384_384035


namespace find_angle_S_l384_384057

variables (A B C D E : Type)
variables (AD BE : straight_line)
variables (AB AC : ℝ)
variables (R S : ℝ)
variables (parallel : (AB ∥ ED))
variables (equal_AB_AC : AB = AC)
variables (angle_ABC : ∠ABC = 30)

theorem find_angle_S
  (h_parallel : AB ∥ ED)
  (h_equal_AB_AC : AB = AC)
  (h_angle_ABC : ∠ABC = 30) :
  ∠ADE = 120 := 
sorry

end find_angle_S_l384_384057


namespace terminating_decimal_count_number_of_terminating_decimals_l384_384666

theorem terminating_decimal_count {n : ℕ} (h1 : 1 ≤ n) (h2 : n ≤ 180) :
  (∃ k, n = 9 * k ∧ 1 ≤ 9 * k ∧ 9 * k ≤ 180) ↔ n ∈ (finset.range 21).image (λ k, 9 * k) :=
by sorry

theorem number_of_terminating_decimals :
  (finset.range 21).card = 20 :=
by sorry

end terminating_decimal_count_number_of_terminating_decimals_l384_384666


namespace scientific_notation_of_avian_influenza_virus_diameter_l384_384181

theorem scientific_notation_of_avian_influenza_virus_diameter :
  (0.000000102 : ℝ) = 1.02 * 10 ^ (-7) :=
by
  sorry

end scientific_notation_of_avian_influenza_virus_diameter_l384_384181


namespace trigonometric_form_of_purely_imaginary_l384_384994

-- Definitions from conditions
def a : ℝ := 0
def b : ℝ := -3
def z : ℂ := complex.mk a b

-- Correct answer from solution
def correct_answer : ℂ := 3 * (complex.cos (3 * real.pi / 2) + complex.I * complex.sin (3 * real.pi / 2))

-- Proof statement
theorem trigonometric_form_of_purely_imaginary : z = correct_answer :=
  sorry

end trigonometric_form_of_purely_imaginary_l384_384994


namespace terminating_decimals_count_l384_384719

theorem terminating_decimals_count :
  ∃ (count : ℕ), count = 60 ∧
  ∀ (n : ℕ), 1 ≤ n ∧ n ≤ 180 →
  (∃ m : ℕ, m * 180 = n * (2^2 * 5)) → 60 = count := sorry

end terminating_decimals_count_l384_384719


namespace assignment_count_l384_384588

noncomputable def num_assignment_methods : Nat := 46

theorem assignment_count :
  ∃ (A B C D : Type) (U1 U2 U3 : Type), 
  (A ≠ B) ∧ (U1 ∧ U2 ∧ U3) ∧ 
  (∃ (assign : A → B → C → D → (U1 ∪ U2 ∪ U3)),
    (assign A ≠ assign B) ∧
    (assign C ≠ assign D) ∧
    (assign A ∈ U1 ∪ U2 ∪ U3) ∧
    (assign B ∈ U1 ∪ U2 ∪ U3) ∧
    (assign C ∈ U1 ∪ U2 ∪ U3) ∧
    (assign D ∈ U1 ∪ U2 ∪ U3)) :=
sorry

end assignment_count_l384_384588


namespace possible_values_l384_384091

-- Define the matrix M and the complex entries a, b, c, d
variables {a b c d : ℂ}

-- Conditions given: M^2 = 2I and abd = 2
def matrix (a b c d : ℂ) := ![![a, b, d], ![b, c, a], ![d, a, c]]

def M := matrix a b c d

-- Define the identity matrix in 3x3
def I := ![
  ![1, 0, 0],
  ![0, 1, 0],
  ![0, 0, 1]
]

noncomputable def condition1 := (M ⬝ M = 2 • I)
noncomputable def condition2 := (a * b * d = 2)

-- Define the goal: Possible values of a^3 + b^3 + c^3 + d^3
theorem possible_values : 
  condition1 ∧ condition2 →
    (∃ x : ℂ, x ∈ ({6 + Complex.sqrt 2, 6 - Complex.sqrt 2} : set ℂ) 
      ∧ a^3 + b^3 + c^3 + d^3 = x) :=
sorry

end possible_values_l384_384091


namespace joint_purchases_popular_l384_384239

-- Define the conditions stating what makes joint purchases feasible
structure Conditions where
  cost_saving : Prop  -- Joint purchases allow significant cost savings.
  shared_overhead : Prop  -- Overhead costs are distributed among all members.
  collective_quality_assessment : Prop  -- Enhanced quality assessment via collective feedback.
  community_trust : Prop  -- Trust within the community encourages honest feedback.

-- Define the proposition stating the popularity of joint purchases
theorem joint_purchases_popular (cond : Conditions) : 
  cond.cost_saving ∧ cond.shared_overhead ∧ cond.collective_quality_assessment ∧ cond.community_trust → 
  Prop := 
by 
  intro h
  sorry

end joint_purchases_popular_l384_384239


namespace probability_genuine_coins_given_weight_condition_l384_384933

/--
Given the following conditions:
- Ten counterfeit coins of equal weight are mixed with 20 genuine coins.
- The weight of a counterfeit coin is different from the weight of a genuine coin.
- Two pairs of coins are selected randomly without replacement from the 30 coins. 

Prove that the probability that all 4 selected coins are genuine, given that the combined weight
of the first pair is equal to the combined weight of the second pair, is 5440/5481.
-/
theorem probability_genuine_coins_given_weight_condition :
  let num_coins := 30
  let num_genuine := 20
  let num_counterfeit := 10
  let pairs_selected := 2
  let pairs_remaining := num_coins - pairs_selected * 2
  let P := (num_genuine / num_coins) * ((num_genuine - 1) / (num_coins - 1)) * ((num_genuine - 2) / pairs_remaining) * ((num_genuine - 3) / (pairs_remaining - 1))
  let event_A_given_B := P / (7 / 16)
  event_A_given_B = 5440 / 5481 := 
sorry

end probability_genuine_coins_given_weight_condition_l384_384933


namespace cone_cross_section_is_equilateral_l384_384519

theorem cone_cross_section_is_equilateral
  (a : ℝ) : 
  (a > 0) → 
  let r := a / 2 in
  let h := a / 2 in
  let s := a / 2 in
  (r = s) → (r = h) → (h = s) → 
  true :=
by
  intros ha r_def h_def s_def r_eq_s r_eq_h h_eq_s
  sorry

end cone_cross_section_is_equilateral_l384_384519


namespace quadrilateral_area_l384_384276

-- Define the conditions given in the problem.
variable (Q : Type) [Quadrilateral Q]
variable (P : Type) [Paper P]
variable (area_of_grey_stripes : ℕ)
variable (area_of_white_stripes : ℕ)
variable (total_area : ℕ)

-- Define the given conditions.
axiom equally_wide_stripes (Q : Type) : (area_of_grey_stripes = area_of_white_stripes)
axiom grey_stripes_area (Q : Type) : (area_of_grey_stripes = 10)
axiom total_quadrilateral_area (Q : Type) : (total_area = area_of_grey_stripes + area_of_white_stripes)

-- Prove the area of the quadrilateral.
theorem quadrilateral_area (Q : Type) [Quadrilateral Q] [Paper P] :
  total_area = 20 :=
by
  -- Given conditions
  have h1 : area_of_grey_stripes = 10 := grey_stripes_area Q
  have h2 : area_of_grey_stripes = area_of_white_stripes := equally_wide_stripes Q

  -- Calculate total area
  calc
    total_area
      = area_of_grey_stripes + area_of_white_stripes : total_quadrilateral_area Q
  ... = 10 + 10 : by rw [h1, h2]
  ... = 20 : rfl

end quadrilateral_area_l384_384276


namespace volume_ratio_l384_384429

-- Definitions of geometric volumes
def volume_cylinder (r h : ℝ) : ℝ := π * r^2 * h
def volume_cone (r h : ℝ) : ℝ := (1 / 3) * π * r^2 * h
def volume_sphere (r : ℝ) : ℝ := (4 / 3) * π * r^3

-- Given conditions
def r_cylinder : ℝ := 9
def h_cylinder : ℝ := 15
def r_cone : ℝ := r_cylinder
def h_cone : ℝ := 5
def r_sphere : ℝ := r_cylinder / 2

-- Volumes calculated from conditions
def V_cylinder : ℝ := volume_cylinder r_cylinder h_cylinder
def V_cone : ℝ := volume_cone r_cone h_cone
def V_sphere : ℝ := volume_sphere r_sphere
def V_combined : ℝ := V_cone + V_sphere

-- Theorem to prove the ratio
theorem volume_ratio : V_combined / V_cylinder = 19 / 90 :=
by
  sorry  -- the proof is omitted

end volume_ratio_l384_384429


namespace more_than_half_sunflower_by_wednesday_l384_384897

def initial_sunflower_seeds (total: ℝ) : ℝ := 0.3 * total
def daily_added_sunflower_seeds (total: ℝ) : ℝ := 0.3 * total
def daily_remaining_sunflower_seeds (initial: ℝ) : ℝ := 0.8 * initial

theorem more_than_half_sunflower_by_wednesday:
  ∀ total : ℝ, total > 0 → 
  let monday_sunflower := initial_sunflower_seeds total in
  let after_tuesday := daily_remaining_sunflower_seeds monday_sunflower + daily_added_sunflower_seeds total in
  let after_wednesday := daily_remaining_sunflower_seeds after_tuesday + daily_added_sunflower_seeds total in
  after_wednesday > total / 2 :=
sorry

end more_than_half_sunflower_by_wednesday_l384_384897


namespace sum_ABC_distinct_digits_l384_384853

def distinct_digits_1_to_9 (a b c d e f h i j : ℕ) : Prop :=
  ∀ x ∈ {a, b, c, d, e, f, h, i, j}, 1 ≤ x ∧ x ≤ 9 ∧
  ∀ y ∈ {a, b, c, d, e, f, h, i, j}, x = y → x = y

theorem sum_ABC_distinct_digits :
  ∃ A B C D E F H I J : ℕ,
  distinct_digits_1_to_9 A B C D E F H I J ∧
  9 * (2 * (A + B + C) + 9) = 45 * (A + B + C) :=
by
  sorry

end sum_ABC_distinct_digits_l384_384853


namespace f_not_in_M_g_in_M_interval_for_g_l384_384362

def M (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, x ≤ y → f x ≤ f y ∨ f x ≥ f y) ∧
  ∃ (a b : ℝ), ∀ y ∈ set.Icc a b, f y ∈ set.Icc (a / 2) (b / 2)

theorem f_not_in_M : ¬M (λ x : ℝ, x + 2 / x) :=
by
  sorry

theorem g_in_M : M (λ x : ℝ, -x^3) :=
by
  sorry

theorem interval_for_g :
  ∃ (a b : ℝ), a = - (real.sqrt 2) / 2 ∧ b = (real.sqrt 2) / 2 ∧ 
  ∀ y ∈ set.Icc a b, (λ x : ℝ, -x^3) y ∈ set.Icc (a / 2) (b / 2) :=
by
  sorry

end f_not_in_M_g_in_M_interval_for_g_l384_384362


namespace flour_bag_comparison_l384_384207

theorem flour_bag_comparison (W : ℝ) : 
    (∃ (W : ℝ), W > 0 ∧ ((W > 1000 → 2 / 3 * W < W - 1000 / 3) ∧ (W = 1000 → 2 / 3 * W = W - 1000 / 3) ∧ (W < 1000 → 2 / 3 * W > W - 1000 / 3))) → false :=
by
  intro h
  cases h with W hW
  cases hW with hWpos hCompare
  sorry

end flour_bag_comparison_l384_384207


namespace circle_radius_l384_384526

-- Definitions based on given conditions
def x (θ : ℝ) : ℝ := 3 * Real.sin θ + 4 * Real.cos θ
def y (θ : ℝ) : ℝ := 4 * Real.sin θ - 3 * Real.cos θ

theorem circle_radius : ∃ r : ℝ, ∀ θ : ℝ, x θ ^ 2 + y θ ^ 2 = r ^ 2 ∧ r = 5 := by
  sorry

end circle_radius_l384_384526


namespace sequence_infinite_pos_neg_l384_384179

theorem sequence_infinite_pos_neg (a : ℕ → ℝ)
  (h : ∀ k : ℕ, a (k + 1) = (k * a k + 1) / (k - a k)) :
  ∃ (P N : ℕ → Prop), (∀ n, P n ↔ 0 < a n) ∧ (∀ n, N n ↔ a n < 0) ∧ 
  (∀ m, ∃ n, n > m ∧ P n) ∧ (∀ m, ∃ n, n > m ∧ N n) := 
sorry

end sequence_infinite_pos_neg_l384_384179


namespace theta_equals_pi_over_4_is_line_l384_384324

theorem theta_equals_pi_over_4_is_line :
  ∀ (r : ℝ), ∀ (θ : ℝ), θ = (π / 4) → (∃ (x y : ℝ), x = r * cos(θ) ∧ y = r * sin(θ) ∧ y = x * tan(θ)) :=
by 
  intros r θ hθ
  use [r * cos θ, r * sin θ]
  split
  { sorry }
  { sorry }

end theta_equals_pi_over_4_is_line_l384_384324


namespace area_triangle_BRS_is_correct_l384_384934

-- Define the point B
structure Point where
  x : ℝ
  y : ℝ

-- Define the given points and conditions
def B : Point := ⟨4, 10⟩

variables (R S : Point)

-- Define the perpendicular condition on lines intersecting at B such that their y-intercepts sum to zero
def perpendicular_lines_intersect_B (R S : Point) :=
  R.y + S.y = 0 ∧ B.x * (R.y - S.y) = 0

-- We should prove the area of triangle BRS is 8 * sqrt 29
theorem area_triangle_BRS_is_correct (h : perpendicular_lines_intersect_B R S) :
  1 / 2 * (√((R.x - S.x) ^ 2 + (R.y - S.y) ^ 2)) * (B.y - ((R.y + S.y) / 2)) = 8 * sqrt 29 :=
sorry

end area_triangle_BRS_is_correct_l384_384934


namespace inverse_matrix_proof_l384_384331

variable (A : Matrix (Fin 2) (Fin 2) ℤ)
variable (B : Matrix (Fin 2) (Fin 2) ℤ)
variable (zeroMatrix : Matrix (Fin 2) (Fin 2) ℤ := ![(0, 0), (0, 0)])

-- Condition: The given matrices
def matrixA := ![(5, -3), (-2, 1)]
def matrixB := ![(-1, -3), (-2, -5)]

-- Property to prove: matrixB is the inverse of matrixA
theorem inverse_matrix_proof : 
  (∀ A : Matrix (Fin 2) (Fin 2) ℤ, A = matrixA) →
  (∀ B : Matrix (Fin 2) (Fin 2) ℤ, B = matrixB) →
  (B ⬝ A = 1) := 
  by sorry

end inverse_matrix_proof_l384_384331


namespace integral_cosine_l384_384650

open Real

theorem integral_cosine : ∫ x in 0..(3 * π / 2), cos x = -1 := 
by
  sorry

end integral_cosine_l384_384650


namespace trailing_zeros_in_2008_factorial_l384_384642

def num_trailing_zeros (n : ℕ) : ℕ :=
  let rec count_factors (n p : ℕ) : ℕ :=
    if n = 0 then 0 else (n / p) + count_factors (n / p) p
  count_factors n 5

theorem trailing_zeros_in_2008_factorial :
  num_trailing_zeros 2008 = 500 :=
by
  sorry

end trailing_zeros_in_2008_factorial_l384_384642


namespace inverse_ratio_l384_384518

-- Define the given function g(x)
def g (x : ℝ) : ℝ := (3 * x - 2) / (x + 4)

-- Define the inverse function g⁻¹(x) in the specified form
def g_inv (x : ℝ) : ℝ := (a * x + b) / (c * x + d)

-- Prove the condition
theorem inverse_ratio (a b c d : ℝ) (h : ∀ x, g (g_inv x) = x) : a / c = -4 :=
  sorry

end inverse_ratio_l384_384518


namespace terminating_decimal_fraction_count_l384_384691

theorem terminating_decimal_fraction_count :
  (finset.filter (λ n : ℕ, 1 ≤ n ∧ n ≤ 180 ∧ 90 ∣ n) (finset.range 181)).card = 20 :=
by sorry

end terminating_decimal_fraction_count_l384_384691


namespace cubes_painted_on_one_side_l384_384127

def is_cube_painted_on_one_side (l w h : ℕ) (cube_size : ℕ) : ℕ :=
  let top_bottom := (l - 2) * (w - 2) * 2
  let front_back := (l - 2) * (h - 2) * 2
  let left_right := (w - 2) * (h - 2) * 2
  top_bottom + front_back + left_right

theorem cubes_painted_on_one_side (l w h cube_size : ℕ) (h_l : l = 5) (h_w : w = 4) (h_h : h = 3) (h_cube_size : cube_size = 1) :
  is_cube_painted_on_one_side l w h cube_size = 22 :=
by
  sorry

end cubes_painted_on_one_side_l384_384127


namespace unit_vector_exists_l384_384653

def vec1 : ℝ × ℝ × ℝ := (2, 3, 1)
def vec2 : ℝ × ℝ × ℝ := (-1, 1, 4)
def magnitude (v : ℝ × ℝ × ℝ) := real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)
def cross_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (v1.2 * v2.3 - v1.3 * v2.2, v1.3 * v2.1 - v1.1 * v2.3, v1.1 * v2.2 - v1.2 * v2.1)

theorem unit_vector_exists :
  ∃ (u : ℝ × ℝ × ℝ), 
    u = let v := cross_product vec1 vec2 in
        let len := magnitude v in
        (v.1 / len, v.2 / len, v.3 / len) ∧
    magnitude u = 1 ∧
    cross_product vec1 u = (0, 0, 0) ∧
    cross_product vec2 u = (0, 0, 0) :=
sorry

end unit_vector_exists_l384_384653


namespace total_integers_at_least_eleven_l384_384055

theorem total_integers_at_least_eleven (n neg_count : ℕ) 
  (h1 : neg_count % 2 = 1)
  (h2 : neg_count ≤ 11) :
  n ≥ 11 := 
sorry

end total_integers_at_least_eleven_l384_384055


namespace terminating_decimal_integers_count_l384_384744

/-- For how many integer values of n between 1 and 180 inclusive does the decimal 
representation of n / 180 terminate? We need to calculate the number of integers n such 
that n includes at least 9 as a factor. The answer is 20. -/
theorem terminating_decimal_integers_count :
  { n : ℤ | 1 ≤ n ∧ n ≤ 180 ∧ 9 ∣ n }.to_finset.card = 20 := 
by sorry

end terminating_decimal_integers_count_l384_384744


namespace number_of_solutions_l384_384124
-- Import the Mathlib library

-- Define the conditions and problem
theorem number_of_solutions (n : ℕ) : 
  let r (n : ℕ) := (1 / 2 : ℚ) * (n + 1) + (1 / 4 : ℚ) * (1 + (-1) ^ n)
  in (∀ x y : ℕ, x + 2 * y = n) ↔ r(n) = (1 / 2 : ℚ) * (n + 1) + (1 / 4 : ℚ) * (1 + (-1) ^ n) :=
sorry

end number_of_solutions_l384_384124


namespace k_20_coloring_connected_l384_384541

open SimpleGraph

theorem k_20_coloring_connected :
  ∃ c : Fin 5, 
  ∀ (K : SimpleGraph (Fin 20)) 
    (colored_K : ∀ e : K.edgeSet, Fin 5),
    (K = completeGraph (Fin 20)) → 
    ¬(K.deleteEdges (colored_K⁻¹' {c})).Disconnected :=
sorry

end k_20_coloring_connected_l384_384541


namespace PUMaC_champion_number_l384_384171

theorem PUMaC_champion_number :
  (let s := { x // 1 ≤ x ∧ x ≤ 2020 } in
   let groups := s.subsets (101) in
   let winners := (λ g : set ℕ, (g.sort Linear.Order.le).head) '' groups in
   let champion := winners.toFinset.sort Linear.Order.le in
   ∃ a b : ℕ, gcd a b = 1 ∧ (champion.sum / (champion.card : ℚ)).num = 2021
    ∧ (champion.sum / (champion.card : ℚ)).denom = 102
    ∧ a + b = 2123) :=
begin
  sorry
end

end PUMaC_champion_number_l384_384171


namespace total_toys_given_l384_384638

theorem total_toys_given (toys_for_boys : ℕ) (toys_for_girls : ℕ) (h1 : toys_for_boys = 134) (h2 : toys_for_girls = 269) : 
  toys_for_boys + toys_for_girls = 403 := 
by 
  sorry

end total_toys_given_l384_384638


namespace sum_of_ages_of_sarahs_friends_l384_384911

noncomputable def sum_of_ages (a b c : ℕ) : ℕ := a + b + c

theorem sum_of_ages_of_sarahs_friends (a b c : ℕ) (h_distinct : ∀ (x y z : ℕ), x ≠ y ∧ y ≠ z ∧ x ≠ z)
  (h_single_digits : ∀ (x y z : ℕ), x < 10 ∧ y < 10 ∧ z < 10)
  (h_product_36 : ∃ (x y : ℕ), x * y = 36 ∧ x ≠ y)
  (h_factor_36 : ∀ (x y z : ℕ), x ∣ 36 ∧ y ∣ 36 ∧ z ∣ 36) :
  ∃ (a b c : ℕ), sum_of_ages a b c = 16 := 
sorry

end sum_of_ages_of_sarahs_friends_l384_384911


namespace distance_traveled_by_light_in_10_seconds_l384_384381

theorem distance_traveled_by_light_in_10_seconds :
  ∃ (a : ℝ) (n : ℕ), (300000 * 10 : ℝ) = a * 10 ^ n ∧ n = 6 :=
sorry

end distance_traveled_by_light_in_10_seconds_l384_384381


namespace reach_desired_state_impossible_l384_384539

-- Defining initial and required states and operations
def token := ℤ

inductive color
| red : color
| blue : color

def position (c : color) : token :=
  match c with
  | color.red   => 0
  | color.blue  => 1

-- Allowed operations
inductive operation
| insert (c : color) : operation
| remove (c : color) : operation

-- Initial configuration: red on the left, blue on the right
def initial_config := [position color.red, position color.blue]

-- Desired configuration: blue on the left, red on the right
def desired_config := [(-1 : token), position color.red]

-- Prove that reaching the desired final state is impossible
theorem reach_desired_state_impossible
  (initial_config = [position color.red, position color.blue])
  (desired_config = [(-1 : token), position color.red])
  (operations : list operation) : 
  sum (map position initial_config) ≠ -1 :=
by
  sorry

end reach_desired_state_impossible_l384_384539


namespace problem_A_problem_B_problem_C_problem_D_l384_384857

variables {a b c : ℝ} -- You may want to use real vectors
variables (AB BC CA : ℝ)
variables (A B C : Real.Angle)

theorem problem_A (h : |a| > |b|) : sin A > sin B :=
sorry

theorem problem_B (h : (a • b) > 0) : (triangle_OBTUSE ABC) := -- you may need to define what a obtuse triangle attibute
sorry

theorem problem_C (h : (a • b) = 0) : right_triangle ABC := 
sorry

theorem problem_D (h : ((b + c - a) • (b + a - c)) = 0) : right_triangle ABC :=
sorry

end problem_A_problem_B_problem_C_problem_D_l384_384857


namespace solve_inequality_l384_384507

theorem solve_inequality (a : ℝ) :
  ∃ S : set ℝ, S = 
    if a > 1 then {x | x < (1 - real.sqrt a) / (a - 1) ∨ x > (1 + real.sqrt a) / (a - 1)} else
    if a = 1 then {x | x > 1 / 2} else
    if 0 < a ∧ a < 1 then {x | (1 - real.sqrt a) / (1 - a) < x ∧ x < (1 + real.sqrt a) / (1 - a)} else
    ∅ :=
by
  sorry

end solve_inequality_l384_384507


namespace joint_purchases_popular_l384_384240

-- Define the conditions stating what makes joint purchases feasible
structure Conditions where
  cost_saving : Prop  -- Joint purchases allow significant cost savings.
  shared_overhead : Prop  -- Overhead costs are distributed among all members.
  collective_quality_assessment : Prop  -- Enhanced quality assessment via collective feedback.
  community_trust : Prop  -- Trust within the community encourages honest feedback.

-- Define the proposition stating the popularity of joint purchases
theorem joint_purchases_popular (cond : Conditions) : 
  cond.cost_saving ∧ cond.shared_overhead ∧ cond.collective_quality_assessment ∧ cond.community_trust → 
  Prop := 
by 
  intro h
  sorry

end joint_purchases_popular_l384_384240


namespace false_proposition_l384_384288

def PropositionA (a b : Line) : Prop :=
  (¬ a ∩ b) → (a ∥ b ∨ a ∥ b)

def PropositionB (a b : Line) (α : Plane) : Prop :=
  (a ∥ b) ↔ (∃ (θ : ℝ), angle_with_plane a α = θ ∧ angle_with_plane b α = θ)

def PropositionC (a b : Line) (β : Plane) : Prop :=
  (a ∥ β) → (a ⊥ b)

def PropositionD (a : Line) (α : Plane) (c : Line) : Prop :=
  (a ∥ c ∧ c ⊆ α) → (a ∥ α)

theorem false_proposition :
  ∃ a b : Line, ∃ α β : Plane, ∃ c : Line, PropositionA a b ∧ ¬ PropositionB a b α ∧ PropositionC a b β ∧ PropositionD a α c :=
by
  sorry

end false_proposition_l384_384288


namespace find_cost_price_l384_384229

-- Given conditions
variables (CP SP1 SP2 : ℝ)
def condition1 : Prop := SP1 = 0.90 * CP
def condition2 : Prop := SP2 = 1.10 * CP
def condition3 : Prop := SP2 - SP1 = 500

-- Prove that CP is 2500 
theorem find_cost_price 
  (CP SP1 SP2 : ℝ)
  (h1 : condition1 CP SP1)
  (h2 : condition2 CP SP2)
  (h3 : condition3 SP1 SP2) : 
  CP = 2500 :=
sorry -- proof not required

end find_cost_price_l384_384229


namespace rectangle_area_correct_l384_384829

noncomputable def rectangle_area (a b : ℝ) : ℝ :=
  (a + b) * (a - b)

theorem rectangle_area_correct (a b : ℝ) :
  (sqrt ((a + b) ^ 2 + (a - b) ^ 2))^2 = (a + b) ^ 2 + (a - b) ^ 2 → 
  rectangle_area a b = a^2 - b^2 :=
by
  sorry

end rectangle_area_correct_l384_384829


namespace find_initial_amount_l384_384973

-- Define the initial amount of money as X
def initial_amount_of_money (X : ℝ) : Prop := 
  (5/7) * X = 245

-- Prove that the initial amount of money is $343
theorem find_initial_amount : ∃ X : ℝ, initial_amount_of_money X ∧ X = 343 :=
by
  use 343
  split
  . unfold initial_amount_of_money
    norm_num
  . norm_num
  . sorry

end find_initial_amount_l384_384973


namespace dividend_correct_l384_384231

theorem dividend_correct (quotient divisor remainder : ℕ) (h1 : quotient = 120) (h2 : divisor = 456) (h3 : remainder = 333) :
  (divisor * quotient + remainder = 55053) :=
by
  rw [h1, h2, h3]
  have h : 456 * 120 + 333 = 54720 + 333 := by rfl
  rw h
  exact rfl

end dividend_correct_l384_384231


namespace product_of_p_r_s_l384_384416

theorem product_of_p_r_s (p r s : ℕ) 
  (h1 : 4^p + 4^3 = 280)
  (h2 : 3^r + 29 = 56) 
  (h3 : 7^s + 6^3 = 728) : 
  p * r * s = 27 :=
by
  sorry

end product_of_p_r_s_l384_384416


namespace exists_subsets_S_l384_384100

open Finset

def exists_two_subsets_with_equal_sum (S : Finset ℕ) : Prop :=
  ∃ (x y u v : ℕ), {x, y} ⊆ S ∧ {u, v} ⊆ S ∧ x + y = u + v ∧ {x, y} ≠ {u, v}

theorem exists_subsets_S (S : Finset ℕ) (h1 : S ⊆ range 25) (h2 : S.card = 10) :
  exists_two_subsets_with_equal_sum S :=
by
  sorry

end exists_subsets_S_l384_384100


namespace ratio_of_capital_l384_384549

variable (C A B : ℝ)
variable (h1 : B = 4 * C)
variable (h2 : B / (A + 5 * C) = 6000 / 16500)

theorem ratio_of_capital : A / B = 17 / 4 :=
by
  sorry

end ratio_of_capital_l384_384549


namespace smallest_number_of_eggs_l384_384573

-- Defining the conditions as hypotheses
def number_of_eggs (c : ℕ) : ℕ := 15 * c - 6

-- Formalizing the statement
theorem smallest_number_of_eggs (c : ℕ) (h : 15 * c - 6 > 150) : 
  c ≥ 11 ∧ number_of_eggs c = 15 * 11 - 6 → number_of_eggs c = 159 :=
by 
  intros h1 h2
  sorry

end smallest_number_of_eggs_l384_384573


namespace basketball_team_selection_l384_384253

noncomputable def count_ways_excluding_twins (n k : ℕ) : ℕ :=
  let total_ways := Nat.choose n k
  let exhaustive_cases := Nat.choose (n - 2) (k - 2)
  total_ways - exhaustive_cases

theorem basketball_team_selection :
  count_ways_excluding_twins 12 5 = 672 :=
by
  sorry

end basketball_team_selection_l384_384253


namespace num_terminating_decimals_l384_384676

theorem num_terminating_decimals (n : ℕ) (h1 : 1 ≤ n ∧ n ≤ 180) :
  (∃ k : ℕ, k = 20) := by
  have h2 : ∀ n, (1 ≤ n ∧ n ≤ 180) → (∃ k, k = (180 / 9)) := sorry
  have h3 : 180 / 9 = 20 := by norm_num
  use 20
  exact h3

end num_terminating_decimals_l384_384676


namespace Q_has_negative_root_l384_384310

def Q (x : ℝ) : ℝ := x^7 + 2 * x^5 + 5 * x^3 - x + 12

theorem Q_has_negative_root : ∃ x : ℝ, x < 0 ∧ Q x = 0 :=
by
  sorry

end Q_has_negative_root_l384_384310


namespace terminating_decimal_count_number_of_terminating_decimals_l384_384665

theorem terminating_decimal_count {n : ℕ} (h1 : 1 ≤ n) (h2 : n ≤ 180) :
  (∃ k, n = 9 * k ∧ 1 ≤ 9 * k ∧ 9 * k ≤ 180) ↔ n ∈ (finset.range 21).image (λ k, 9 * k) :=
by sorry

theorem number_of_terminating_decimals :
  (finset.range 21).card = 20 :=
by sorry

end terminating_decimal_count_number_of_terminating_decimals_l384_384665


namespace math_equivalence_proof_l384_384635

theorem math_equivalence_proof :
  ( (8 / 27 : ℝ) ^ ((-2: ℝ) / 3) + real.logb 5 3 - real.logb 5 15 - (real.sqrt 2 - 1) ^ ((real.log 10 1) : ℝ) = 1 / 4) :=
by
  sorry

end math_equivalence_proof_l384_384635


namespace seven_people_different_rolls_l384_384917

def rolls_different (rolls : Fin 7 -> Fin 6) : Prop :=
  ∀ i : Fin 7, rolls i ≠ rolls ⟨(i + 1) % 7, sorry⟩

def probability_rolls_different : ℚ :=
  (625 : ℚ) / 2799

theorem seven_people_different_rolls (rolls : Fin 7 -> Fin 6) :
  (∃ rolls, rolls_different rolls) ->
  probability_rolls_different = 625 / 2799 :=
sorry

end seven_people_different_rolls_l384_384917


namespace take_home_amount_l384_384992

-- Define conditions
def total_winnings := 50
def tax_rate := 0.20
def processing_fee := 5

-- Define tax amount and total deductions
def tax_amount := total_winnings * tax_rate
def total_deductions := tax_amount + processing_fee

-- Define amount taken home
def amount_taken_home := total_winnings - total_deductions

-- Theorem to prove
theorem take_home_amount : amount_taken_home = 35 :=
by
  sorry

end take_home_amount_l384_384992


namespace pascals_identity_l384_384137

-- Pascal's Identity
theorem pascals_identity (n k : ℕ) : nat.choose n k + nat.choose n (k + 1) = nat.choose (n + 1) (k + 1) := 
by
  sorry

end pascals_identity_l384_384137


namespace geometric_sequence_a3_equals_4_l384_384075

variable {a : ℕ → ℝ}

def is_geometric_sequence (a : ℕ → ℝ) := ∃ r : ℝ, ∀ i, a (i+1) = a i * r

theorem geometric_sequence_a3_equals_4 
    (a_seq : is_geometric_sequence a) 
    (a_6_eq : a 6 = 6)
    (a_9_eq : a 9 = 9) : 
    a 3 = 4 := 
sorry

end geometric_sequence_a3_equals_4_l384_384075


namespace angle_CAD_lt_90_l384_384869

noncomputable def circle (α : Type*) [metric_space α] := set α

variables {α : Type*} [metric_space α]
variables (ω₁ ω₂ : circle α) (A B X Y C D : α)
variables (M N : α) (O₁ O₂ : α)
variables (h1 : A ∈ ω₁) (h2 : A ∈ ω₂) 
variables (h3 : B ∈ ω₁) (h4 : B ∈ ω₂)
variables (h5 : X ∈ ω₁) (h6 : Y ∈ ω₂)
variables (h7 : tangent_line XY ω₁) (h8 : tangent_line XY ω₂) -- assuming tangent_line represents the tangency
variables (h9 : closer_to B XY) -- assuming closer_to represents the distance condition
variables (h10 : reflective B X C) (h11 : reflective B Y D) -- reflective means reflection

-- To be proved
theorem angle_CAD_lt_90 : angle CAD < 90 := 
sorry

end angle_CAD_lt_90_l384_384869


namespace integer_solutions_of_equation_l384_384655

theorem integer_solutions_of_equation :
  ∀ x y : ℤ, 2 * x^3 + x * y - 7 = 0 ↔ (x = -7 ∧ y = -99) ∨ (x = -1 ∧ y = -9) ∨ (x = 1 ∧ y = 5) ∨ (x = 7 ∧ y = -97) := by 
  sorry

end integer_solutions_of_equation_l384_384655


namespace sum_of_solutions_eq_zero_final_sum_of_solutions_l384_384985

theorem sum_of_solutions_eq_zero (x : ℝ) (h : (6 * x) / 18 = 8 / x) : x = 24 ∨ x = -24 :=
by
  have hx : x ≠ 0 := by sorry
  have : (1 / 3) * x = 8 := by sorry
  have : x = 24 := by sorry
  have : x = -24 := by sorry

theorem final_sum_of_solutions : ∑ x in {24, -24}, x = 0 :=
by
  simp


end sum_of_solutions_eq_zero_final_sum_of_solutions_l384_384985


namespace find_a_l384_384777

theorem find_a (a : ℝ) (l1 : ∀ x y, ax - y - 2 = 0) (l2 : ∀ x y, (a + 2)x - y + 1 = 0) 
  (perp : ∀ m1 m2, m1 * m2 = -1) : a = -1 :=
by
  sorry

end find_a_l384_384777


namespace part1_part2_l384_384800

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := log (a * x + 1 / 2) + 2 / (2 * x + 1)

theorem part1 (a : ℝ) (h : a > 0) : 
  (∀ x > 0, deriv (f x a) ≥ 0) ↔ a ≥ 2 := sorry

theorem part2 : ∃ a > 0, (∀ x > 0, f x a ≥ 1) ∧ (∃ x > 0, f x a = 1) := sorry

end part1_part2_l384_384800


namespace only_n_divides_2_to_n_minus_1_l384_384654

theorem only_n_divides_2_to_n_minus_1 (n : ℕ) (h1 : n > 0) : n ∣ (2^n - 1) ↔ n = 1 :=
by
  sorry

end only_n_divides_2_to_n_minus_1_l384_384654


namespace integer_roots_count_l384_384513

theorem integer_roots_count (b c d e f : ℚ) :
  ∃ (n : ℕ), (n = 0 ∨ n = 1 ∨ n = 2 ∨ n = 4 ∨ n = 5) ∧
  (∃ (r : ℕ → ℤ), ∀ i, i < n → (∀ z : ℤ, (∃ m, z = r m) → (z^5 + b * z^4 + c * z^3 + d * z^2 + e * z + f = 0))) :=
sorry

end integer_roots_count_l384_384513


namespace proof_of_equilateral_triangle_l384_384128

noncomputable theory

open_locale classical

variables {A B C A₁ B₁ C₁ P Q : Type} 
  [metric_space A] [metric_space B] [metric_space C]
  [metric_space A₁] [metric_space B₁] [metric_space C₁]
  [metric_space P] [metric_space Q]

-- Assume required configurations and constructions
variables (triangle_ABC : is_triangle A B C)
          (triangle_ABC1 : is_equilateral_triangle A B C₁)
          (triangle_AB1C : is_equilateral_triangle A B₁ C)
          (triangle_A1BC : is_equilateral_triangle A₁ B C)
          (P_is_midpoint : midpoint A₁ B₁ = P)
          (Q_is_midpoint : midpoint A₁ C₁ = Q)

def is_equilateral_triangle (A P Q : Type) [metric_space A] [metric_space P] [metric_space Q] : Prop := sorry

theorem proof_of_equilateral_triangle :
  is_equilateral_triangle A P Q := 
sorry

end proof_of_equilateral_triangle_l384_384128


namespace subset_count_l384_384634

/-- 
Let X be a subset of {1, 2, 3, 5, 6}. We want to prove the number of subsets X such that
{1, 2, 3} is a subset of X.
-/
theorem subset_count : 
  let S := {1, 2, 3, 5, 6} in
  let conditions := {1, 2, 3} in 
  let count := (finset.powerset_len 2 (finset.ofList [5, 6])).card in
  count = 4 :=
by
  let S := {1, 2, 3, 5, 6}
  let conditions := {1, 2, 3}
  let count := (finset.powerset_len 2 (finset.ofList [5, 6])).card
  have h : count = 4 := 
    sorry -- The detailed proof will go here.
  exact h

end subset_count_l384_384634


namespace simplify_complex_fraction_l384_384918

-- Define the complex numbers involved
def numerator := 3 + 4 * Complex.I
def denominator := 5 - 2 * Complex.I

-- Define what we need to prove: the simplified form
theorem simplify_complex_fraction : 
    (numerator / denominator : Complex) = (7 / 29) + (26 / 29) * Complex.I := 
by
  -- Proof is omitted here
  sorry

end simplify_complex_fraction_l384_384918


namespace mallory_total_expenses_l384_384156
noncomputable theory

-- Definitions from the conditions
def fuel_tank_cost : ℝ := 45
def miles_per_tank : ℝ := 500
def journey_distance : ℝ := 2000
def fuel_increment : ℝ := 5
def food_factor : ℝ := 3 / 5
def hotel_cost_per_night : ℝ := 80
def hotel_nights : ℕ := 3
def unexpected_maintenance_cost : ℝ := 120
def activities_cost : ℝ := 50

-- Theorem statement to be proven
theorem mallory_total_expenses : 
  let total_fuel_cost := 
      (fuel_tank_cost + (fuel_tank_cost + fuel_increment) + 
      (fuel_tank_cost + 2 * fuel_increment) + 
      (fuel_tank_cost + 3 * fuel_increment))
  let food_cost := food_factor * total_fuel_cost
  let hotel_cost := hotel_nights * hotel_cost_per_night
  let extra_expenses := unexpected_maintenance_cost + activities_cost
  let total_expenses := total_fuel_cost + food_cost + hotel_cost + extra_expenses
  in total_expenses = 746 :=
by
  sorry

end mallory_total_expenses_l384_384156


namespace second_number_added_is_5_l384_384966

theorem second_number_added_is_5
  (x : ℕ) (h₁ : x = 3)
  (y : ℕ)
  (h₂ : (x + 1) * (x + 13) = (x + y) * (x + y)) :
  y = 5 :=
sorry

end second_number_added_is_5_l384_384966


namespace temperature_celsius_range_l384_384205

theorem temperature_celsius_range (C : ℝ) :
  (∀ C : ℝ, let F_approx := 2 * C + 30;
             let F_exact := (9 / 5) * C + 32;
             abs ((2 * C + 30 - ((9 / 5) * C + 32)) / ((9 / 5) * C + 32)) ≤ 0.05) →
  (40 / 29) ≤ C ∧ C ≤ (360 / 11) :=
by
  intros h
  sorry

end temperature_celsius_range_l384_384205


namespace required_line_eq_l384_384326

theorem required_line_eq :
  (∃ (P : ℝ × ℝ), (2 * P.1 + 3 * P.2 + 1 = 0) ∧ (P.1 - 3 * P.2 + 4 = 0)) →
  (∃ (m : ℝ), (∃ (P : ℝ × ℝ), 4 * P.1 - 3 * P.2 + m = 0) ∧ (3 * P.1 + 4 * P.2 - 7 = 0)) →
  (∃ (P : ℝ × ℝ), (4 * P.1 - 3 * P.2 + 1 = 0)) :=
by
  sorry

end required_line_eq_l384_384326


namespace number_of_sequences_l384_384885

theorem number_of_sequences (n k : ℕ) (h : 1 ≤ k ∧ k < n) : 
  (∑ s in {s : Fin k → ℕ | (∀ i, s i > 0) ∧ (∑ i in Finset.univ, s i = n)}, 1) = nat.choose (n - 1) (k - 1) := 
sorry

end number_of_sequences_l384_384885


namespace Annette_Caitlin_total_weight_l384_384292

variable (A C S : ℕ)

-- Conditions
axiom cond1 : C + S = 87
axiom cond2 : A = S + 8

-- Theorem
theorem Annette_Caitlin_total_weight : A + C = 95 := by
  sorry

end Annette_Caitlin_total_weight_l384_384292


namespace line_parallel_to_plane_l384_384344

-- Defining conditions
def vector_a : ℝ × ℝ × ℝ := (1, -1, 3)
def vector_n : ℝ × ℝ × ℝ := (0, 3, 1)

-- Lean theorem statement
theorem line_parallel_to_plane : 
  let ⟨a1, a2, a3⟩ := vector_a;
  let ⟨n1, n2, n3⟩ := vector_n;
  a1 * n1 + a2 * n2 + a3 * n3 = 0 :=
by 
  -- Proof omitted
  sorry

end line_parallel_to_plane_l384_384344


namespace hyperbola_asymptotes_l384_384025

open Real

noncomputable def hyperbola_equation_exists (b : ℝ) (hb : b > 0) : Prop :=
  ∃ (asymptotes : String), (asymptotes = "3x ± 4y = 0")

theorem hyperbola_asymptotes {b : ℝ} (hb : b > 0) :
  hyperbola_equation_exists b hb :=
by {
  -- Conditions for the hyperbola and parabola are given
  let a := 4,
  let c := 5,
  have h1 : c^2 = a^2 + b^2 := by linarith,
  have h2 : (focus : ℝ × ℝ) = (5, 0),
  -- The correct computations lead to b = 3 and then we obtain the asymptotes
  have asymptote_equations := "3x ± 4y = 0",
  use asymptote_equations,
  sorry
}

end hyperbola_asymptotes_l384_384025


namespace terminating_decimals_l384_384707

theorem terminating_decimals (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 180) :
  (∃ k : ℕ, k = 20 ∧ ∀ m : ℕ, 1 ≤ m ∧ m ≤ 180 → (decimal_terminates (m / 180) ↔ m % 9 = 0)) :=
by {
  sorry
}

def decimal_terminates (x : ℚ) : Prop :=
  sorry

end terminating_decimals_l384_384707


namespace area_given_binomial_coefficient_l384_384964

noncomputable def area_closed_figure : ℝ :=
  ∫ x in 0..1, x^(1/3) - x^2

theorem area_given_binomial_coefficient : area_closed_figure = 5 / 12 :=
by
  -- The sum of the coefficients condition helps us determine that a = 1 / 3
  have h : (1 + 1 / (1 / 3)) ^ 5 = 1024 := by sorry
  -- Now we calculate the area of the closed figure
  calc
    ∫ x in 0..1, x^(1/3) - x^2 
        = (3 / 4 * 1^(4/3) - 1/3 * 1^3) - (3 / 4 * 0^(4/3) - 1/3 * 0^3) : by sorry
    ... = 3 / 4 - 1 / 3 : by sorry
    ... = 5 / 12 : by sorry

end area_given_binomial_coefficient_l384_384964


namespace triangle_area_proof_l384_384528

noncomputable def area_of_triangle (a b c : ℝ) (alpha beta gamma : ℝ) : ℝ :=
  let s := (a + b + c) / 2 in
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem triangle_area_proof :
  ∃ (A B C : ℝ) (alpha beta : ℝ),
    (A = 5) ∧
    (2 * beta = alpha) ∧
    (C - B = 2) ∧
    area_of_triangle A B C alpha beta (π - alpha - beta) = 3.75 * Real.sqrt 7 :=
begin
  sorry
end

end triangle_area_proof_l384_384528


namespace abs_conditions_iff_l384_384357

theorem abs_conditions_iff (x y : ℝ) :
  (|x| < 1 ∧ |y| < 1) ↔ (|x + y| + |x - y| < 2) :=
by
  sorry

end abs_conditions_iff_l384_384357


namespace irrational_roots_of_quadratic_l384_384414

theorem irrational_roots_of_quadratic (p q : ℤ) (h1 : p % 2 = 1) (h2 : q % 2 = 1) (h3 : p^2 - 2*q ≥ 0) :
  ∀ x, ¬ ∃ r : ℚ, x = -p + Real.sqrt (p^2 - 2*q) ∨ x = -p - Real.sqrt (p^2 - 2*q) ∧ r = (x : ℝ) := sorry

end irrational_roots_of_quadratic_l384_384414


namespace smallest_absolute_difference_of_products_of_roots_l384_384471

noncomputable def g (x : ℝ) : ℝ := x^4 + 10 * x^3 + 35 * x^2 + 50 * x + 24

theorem smallest_absolute_difference_of_products_of_roots :
  ∃ w1 w2 w3 w4 : ℝ,
    (polynomial.map (algebra_map ℝ ℂ) (polynomial.C 24 * polynomial.expand ℝ 2 g)).roots = {w1, w2, w3, w4}.multiset ∧
    (∃ a b c d : ℕ, {a, b, c, d} = {1, 2, 3, 4} ∧ |w_a w_b - w_c w_d| = 0) := sorry

end smallest_absolute_difference_of_products_of_roots_l384_384471


namespace exists_even_numbers_in_same_subset_l384_384530

theorem exists_even_numbers_in_same_subset
  (n k : ℕ) (M : Finset ℕ)
  (M_part : Π i : ℕ, i ∈ Finset.range (k + 1) → Finset ℕ)
  (part_cond : ∀ i j : ℕ, i ≠ j → M_part i ∩ M_part j = ∅)
  (union_cond : M = Finset.bind (Finset.range (k + 1)) (λ i, M_part i))
  (card_M : M.card = 2 * n)
  (n_cond : n ≥ k^3 + k) :
  ∃ (i j : ℕ) (even_nums : Fin k+1 → ℕ),
    (∀ idx, even_nums idx % 2 = 0) ∧
    (∀ idx1 idx2, M_part i (even_nums idx1) ∧ M_part i (even_nums idx2)) ∧
    (∀ idx1 idx2, M_part j (even_nums idx1 - 1) ∧ M_part j (even_nums idx2 - 1)) :=
sorry

end exists_even_numbers_in_same_subset_l384_384530


namespace isabella_paint_area_l384_384085

-- Lean 4 statement for the proof problem based on given conditions and question:
theorem isabella_paint_area :
  let length := 15
  let width := 12
  let height := 9
  let door_and_window_area := 80
  let number_of_bedrooms := 4
  (2 * (length * height) + 2 * (width * height) - door_and_window_area) * number_of_bedrooms = 1624 :=
by
  sorry

end isabella_paint_area_l384_384085


namespace terminating_decimal_count_number_of_terminating_decimals_l384_384673

theorem terminating_decimal_count {n : ℕ} (h1 : 1 ≤ n) (h2 : n ≤ 180) :
  (∃ k, n = 9 * k ∧ 1 ≤ 9 * k ∧ 9 * k ≤ 180) ↔ n ∈ (finset.range 21).image (λ k, 9 * k) :=
by sorry

theorem number_of_terminating_decimals :
  (finset.range 21).card = 20 :=
by sorry

end terminating_decimal_count_number_of_terminating_decimals_l384_384673


namespace function_has_root_l384_384122

noncomputable def f (n : ℕ) (a b : ℕ → ℝ) (x : ℝ) : ℝ :=
  ∑ i in finset.range (n + 1), (a i * real.sin ((i + 1 : ℕ) * x) + b i * real.cos ((i + 1 : ℕ) * x))

theorem function_has_root (n : ℕ) (a b : ℕ → ℝ) :
  ∃ x ∈ set.Icc 0 (2 * real.pi), f n a b x = 0 :=
begin
  sorry
end

end function_has_root_l384_384122


namespace candy_bars_calculation_l384_384889

theorem candy_bars_calculation (f : ℕ) (b : ℕ) (j : ℕ) : 
    f = 12 ∧ 
    b = f + 6 ∧ 
    j = 10 * (f + b) → 
    0.4 * (j ^ 2) = 36000 :=
by
  intro h
  sorry

end candy_bars_calculation_l384_384889


namespace line_through_points_l384_384379

theorem line_through_points :
  ∀ x y : ℝ, (∃ t : ℝ, (x, y) = (2 * t, -3 * (1 - t))) ↔ (x / 2) - (y / 3) = 1 :=
by
  sorry

end line_through_points_l384_384379


namespace solve_fraction_equation_l384_384922

theorem solve_fraction_equation (x : ℝ) :
  (1 / (x + 10) + 1 / (x + 8) = 1 / (x + 11) + 1 / (x + 7)) ↔ x = -9 :=
by {
  sorry
}

end solve_fraction_equation_l384_384922


namespace suraj_new_average_l384_384234

theorem suraj_new_average (A : ℝ) (avg_condition : A = 20)
  (runs_condition : 14 * A + 140 = 15 * (A + 8)) : 
  (15 * (A + 8)) / 15 = 28 := 
by
  rw [avg_condition, runs_condition]
  sorry

end suraj_new_average_l384_384234


namespace number_of_possible_SC_values_l384_384483

open Finset

variable (𝒞 : Set ℕ)
variable (n a d : ℕ)

def sum_arithmetic_sequence (n a d : ℕ) : ℕ :=
  (n * (2 * a + (n - 1) * d)) / 2

noncomputable def count_possible_sums : ℕ :=
  let S_min := sum_arithmetic_sequence 75 10 1
  let S_max := sum_arithmetic_sequence 75 126 1
  S_max - S_min + 1

theorem number_of_possible_SC_values 
  (𝒞 : Finset ℕ) 
  (h₁ : 𝒞.card = 75) 
  (h₂ : ∀ x ∈ 𝒞, 10 ≤ x ∧ x ≤ 200) :
  count_possible_sums = 8476 := by
  sorry

end number_of_possible_SC_values_l384_384483


namespace pythagorean_theorem_for_orthogonal_random_variables_l384_384102

noncomputable def orthogonal_random_variables (n : ℕ) (ξ : ℕ → ℝ) : Prop :=
∀ i j, i ≠ j → E (ξ i * ξ j) = 0

theorem pythagorean_theorem_for_orthogonal_random_variables 
  (n : ℕ) (ξ : ℕ → ℝ) (h_orthogonal : orthogonal_random_variables n ξ) :
  ‖∑ i in finset.range n, ξ i‖^2 = ∑ i in finset.range n, ‖ξ i‖^2 := by
  sorry

end pythagorean_theorem_for_orthogonal_random_variables_l384_384102


namespace minimum_value_l384_384766

-- Define the geometric sequence and its conditions
variable (a : ℕ → ℝ)
variable (q : ℝ)
variable (positive : ∀ n, 0 < a n)
variable (geometric_seq : ∀ n, a (n+1) = q * a n)
variable (condition1 : a 6 = a 5 + 2 * a 4)
variable (m n : ℕ)
variable (condition2 : ∀ m n, sqrt (a m * a n) = 2 * a 1 → a m = a n)

-- Prove that the minimum value of 1/m + 9/n is 4
theorem minimum_value : m + n = 4 → (∀ x y : ℝ, (0 < x ∧ 0 < y) → (1 / x + 9 / y) ≥ 4) :=
sorry

end minimum_value_l384_384766


namespace final_price_jacket_l384_384297

-- Defining the conditions as per the problem
def original_price : ℚ := 250
def first_discount_rate : ℚ := 0.40
def second_discount_rate : ℚ := 0.15
def tax_rate : ℚ := 0.05

-- Defining the calculation steps
def first_discounted_price : ℚ := original_price * (1 - first_discount_rate)
def second_discounted_price : ℚ := first_discounted_price * (1 - second_discount_rate)
def final_price_inclusive_tax : ℚ := second_discounted_price * (1 + tax_rate)

-- The proof problem statement
theorem final_price_jacket : final_price_inclusive_tax = 133.88 := sorry

end final_price_jacket_l384_384297


namespace intersection_A_B_l384_384463

-- Define the set A
def A : Set ℕ := {1, 2, 3, 4, 5}

-- Define the set B as the set of natural numbers greater than 2.5
def B : Set ℕ := {x : ℕ | 2 * x > 5}

-- Prove that the intersection of A and B is {3, 4, 5}
theorem intersection_A_B : A ∩ B = {3, 4, 5} :=
by sorry

end intersection_A_B_l384_384463


namespace tetrahedron_ratio_l384_384084

theorem tetrahedron_ratio (A B C D O A1 B1 C1 D1 : Point)
  (hA1 : line_through A O ∩ face B C D = A1)
  (hB1 : line_through B O ∩ face A C D = B1)
  (hC1 : line_through C O ∩ face A B D = C1)
  (hD1 : line_through D O ∩ face A B C = D1)
  (h_ratio_eq : AO / A1O = k ∧
                BO / B1O = k ∧
                CO / C1O = k ∧
                DO / D1O = k) :
  k = 3 := 
sorry

end tetrahedron_ratio_l384_384084


namespace solution_l384_384411

noncomputable def problem_statement : Prop :=
  ∃ (x y : ℝ), (x > 0 ∧ y > 0) ∧ (6 * x^2 + 18 * x * y = 2 * x^3 + 3 * x^2 * y^2) ∧ x = (3 + Real.sqrt 153) / 4

theorem solution : problem_statement :=
by
  sorry

end solution_l384_384411


namespace find_s_value_l384_384398

noncomputable def s_value (r s : ℝ) : Prop :=
  1 < r ∧ r < s ∧ (1 / r + 1 / s = 1) ∧ (r * s = 15 / 4) ∧ (s = (15 + real.sqrt 15) / 8)

-- Main theorem statement
theorem find_s_value (r s : ℝ) (h : s_value r s) : s = (15 + real.sqrt 15) / 8 :=
by
  sorry

end find_s_value_l384_384398


namespace graph_is_line_segment_l384_384951

def f (x : ℝ) : ℝ := 3 * x - 1

theorem graph_is_line_segment : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 5 → ∃ y1 y2 x1 x2 : ℝ, f x1 = y1 ∧ f x2 = y2 ∧ 1 ≤ x1 ∧ x1 ≤ 5 ∧ 1 ≤ x2 ∧ x2 ≤ 5 ∧ (x1, y1) ∈ set.image (λ x, (x, f x)) (set.Icc 1 5) ∧ (x2, y2) ∈ set.image (λ x, (x, f x)) (set.Icc 1 5) := by
  trivial
sorry

end graph_is_line_segment_l384_384951


namespace AB_l384_384442

-- Initialize the problem context
variables {A B C H M N B' : Type} 
variables (ABC : Triangle A B C) (acute_ABC : acute ABC)
variables (BH_altitude : Altitude BH) 
variables (M_mid_AH : Midpoint M A H) (N_mid_CH : Midpoint N C H)
variables (Omega : Circumcircle B M N) (BBp_diameter : Diameter BB')

-- The proof statement
theorem AB'_eq_CB' (h1 : Triangle.is_acute ABC) (h2 : Altitude BH)
  (h3 : Midpoint M A H) (h4 : Midpoint N C H) 
  (h5 : Circumcircle B M N) (h6 : Diameter BB') : 
  Distance A B' = Distance C B' := 
by
  sorry

end AB_l384_384442


namespace terminating_decimal_count_l384_384752

theorem terminating_decimal_count : 
  (Finset.filter (λ n, n % 3 = 0) (Finset.range 181)).card = 60 := by
  sorry

end terminating_decimal_count_l384_384752


namespace P_on_AC_l384_384058

-- Definitions of all points and line properties 
variable (O A B C D E F P : Point)
variable (l_circle s_circle : Circle)
variable (tangent_AC : Line)

-- Conditions in the problem
axiom H1 : l_circle.center = O
axiom H2 : s_circle.center = O
axiom H3 : OnCircle A l_circle
axiom H4 : OnCircle C l_circle
axiom H5 : TangentOn A s_circle tangent_AC
axiom H6 : OnCircle B s_circle
axiom H7 : OnLine B C tangent_AC
axiom H8 : midpoint D A B
axiom H9 : OnCircle E s_circle
axiom H10 : SecondIntersection A E s_circle F
axiom H11 : PerpBisectorIntersection CE DF P
axiom H12 : OnLine P tangent_AC

theorem P_on_AC : OnLine P tangent_AC := 
  by sorry

end P_on_AC_l384_384058


namespace minimum_g_l384_384508

noncomputable def g (A B C D X : Point) : ℝ :=
  dist A X + dist B X + dist C X + dist D X

theorem minimum_g (A B C D : Point)
  (hAD : dist A D = 30) (hBC : dist B C = 30)
  (hAC : dist A C = 46) (hBD : dist B D = 46)
  (hAB : dist A B = 50) (hCD : dist C D = 50) :
  ∃ X : Point, g A B C D X = 4 * Real.sqrt 628 :=
sorry

end minimum_g_l384_384508


namespace turban_price_l384_384033

theorem turban_price (
  (T : ℝ) (total_salary : ℝ) (term_salary : ℝ)
  (H1 : total_salary = 90 + T)
  (H2 : term_salary = 55 + T)) 
  (leave_duration_ratio : ℝ := 3/4)
  (H3 : total_salary * leave_duration_ratio = term_salary) :
  T = 50 :=
by 
  sorry

end turban_price_l384_384033


namespace abs_inequality_interval_notation_l384_384315

variable (x : ℝ)

theorem abs_inequality_interval_notation :
  {x : ℝ | |x - 1| < 1} = {x : ℝ | 0 < x ∧ x < 2} :=
by
  sorry

end abs_inequality_interval_notation_l384_384315


namespace imaginary_part_of_conjugate_equals_two_l384_384355

def imaginary_unit : ℂ := complex.I
def complex_number : ℂ := 1 - 2 * complex.I

theorem imaginary_part_of_conjugate_equals_two : complex.im (conj complex_number) = 2 := by
  sorry

end imaginary_part_of_conjugate_equals_two_l384_384355


namespace polygon_name_and_perimeter_l384_384236

-- Define the conditions
def regular_polygon (n : ℕ) (side_length perimeter : ℕ) : Prop :=
  perimeter = n * side_length

axiom length_of_side (P : ℕ) : (P / 5) = 25

-- Define the goal
theorem polygon_name_and_perimeter :
  ∃ (n : ℕ) (P : ℕ), regular_polygon n 25 P ∧ n = 5 ∧ P = 125 :=
by
  use 5, 125
  split
  . unfold regular_polygon
    simp
  . split
    . refl
    . refl
  . sorry

end polygon_name_and_perimeter_l384_384236


namespace tan_shift_symmetric_l384_384950

theorem tan_shift_symmetric :
  let f (x : ℝ) := Real.tan (2 * x + Real.pi / 6)
  let g (x : ℝ) := f (x + Real.pi / 6)
  g (Real.pi / 4) = 0 ∧ ∀ x, g (Real.pi / 2 - x) = -g (Real.pi / 2 + x) :=
by
  sorry

end tan_shift_symmetric_l384_384950


namespace sum_non_visible_faces_l384_384139

theorem sum_non_visible_faces (a : Fin 8 → Fin 6) 
  (h_sum_opposite_faces : ∀ i : Fin 6, ∀ j : Fin 6, i ≠ j → i + j = 7) 
  (h_sum_all_faces : ∑ i in Finset.univ, ∑ j : Fin 6, j = 168) 
  : ∑ i : Fin 8, (7 - a i) = 56 :=
sorry

end sum_non_visible_faces_l384_384139


namespace maximum_ratio_l384_384886

-- Defining the conditions
def two_digit_positive_integer (n : ℕ) : Prop := n >= 10 ∧ n < 100

-- Proving the main theorem
theorem maximum_ratio (x y : ℕ) (hx : two_digit_positive_integer x) (hy : two_digit_positive_integer y) (h_sum : x + y = 100) : 
  ∃ m, m = 9 ∧ ∀ r, r = x / y → r ≤ 9 := sorry

end maximum_ratio_l384_384886


namespace variance_identification_l384_384446

-- defining sample variance formula
def sample_variance (n : ℕ) (x : ℕ → ℝ) (mean : ℝ) : ℝ :=
  (1 / n) * (Finset.sum (Finset.range n) (λ i, (x i - mean) ^ 2))

-- given condition
def given_variance_formula : ℝ :=
  sample_variance 10 
    (λ i, match i with
          | 0 => x_1 | 1 => x_2 | 2 => x_3  -- dummy values for illustration
          -- add all values until x_9 to denote x_0 through x_10 respectively
          | 9 => x_10
          | _ => 0
          end)
    20

-- proof statement problem (not the actual proof)
theorem variance_identification :
  (sample_variance 10 (λ i, x_i) 20 = given_variance_formula) →
  10 = 10 ∧ 20 = 20 :=
sorry

end variance_identification_l384_384446


namespace finite_M_l384_384099

open Matrix

variables {n : Type*} [DecidableEq n] [Fintype n] [Nonempty n]
variables (M : Set (Matrix n n ℝ))

-- Conditions
def I_in_M : Prop := (1 : Matrix n n ℝ) ∈ M

def AB_or_negAB_in_M (A B : Matrix n n ℝ) (hA : A ∈ M) (hB : B ∈ M) : Prop :=
(AB ∈ M ∧ -AB ∉ M) ∨ (AB ∉ M ∧ -AB ∈ M)

def AB_or_negBA_condition (A B : Matrix n n ℝ) (hA : A ∈ M) (hB : B ∈ M) : Prop :=
(AB = BA) ∨ (AB = -BA)

def exists_B_negAB (A : Matrix n n ℝ) (hA : A ∈ M) (hA_ne : A ≠ 1) : Prop :=
∃ B ∈ M, AB = -BA

-- Problem Statement
theorem finite_M : I_in_M M ∧
(∀ A B, A ∈ M → B ∈ M → AB_or_negAB_in_M M A B)
∧ (∀ A B, A ∈ M → B ∈ M → AB_or_negBA_condition M A B)
∧ (∀ A, A ∈ M → A ≠ 1 → exists_B_negAB M A)
→ Fintype.card M ≤ Fintype.card n :=
sorry

end finite_M_l384_384099


namespace terminating_decimal_count_number_of_terminating_decimals_l384_384667

theorem terminating_decimal_count {n : ℕ} (h1 : 1 ≤ n) (h2 : n ≤ 180) :
  (∃ k, n = 9 * k ∧ 1 ≤ 9 * k ∧ 9 * k ≤ 180) ↔ n ∈ (finset.range 21).image (λ k, 9 * k) :=
by sorry

theorem number_of_terminating_decimals :
  (finset.range 21).card = 20 :=
by sorry

end terminating_decimal_count_number_of_terminating_decimals_l384_384667


namespace sum_coordinates_l384_384902

-- Definition of the points C and D
def C := (3, y) : ℝ × ℝ
def D := (3, -y) : ℝ × ℝ

-- Define the theorem to prove the sum of the coordinates of C and D equals 6
theorem sum_coordinates (y : ℝ) : (C.1 + C.2 + D.1 + D.2) = 6 :=
by
  -- Start proof
  sorry

end sum_coordinates_l384_384902


namespace medians_intersect_at_single_point_l384_384905

/-
Define what a tetrahedron is, denoted here by vertices A, B, C, and D
-/

structure Tetrahedron :=
(A B C D : Point)

def is_centroid (G : Point) (ABC : Triangle) : Prop :=
  ∃ M N, ((M = midpoint ABC.ABC.A ABC.ABC.B) ∧ (N = midpoint ABC.ABC.ABC.C) ∧
           G = (1 / 3 * ABC.ABC.A + 1 / 3 * ABC.ABC.B + 1 / 3 * ABC.ABC.C))

def median (T : Tetrahedron) (G : Point) (v : T.ABC.ABC.A | v : T.ABC.B | v : T.ABC.C | v : T.ABC.D) : LineSegment :=
  sorry -- definition of a median connecting v to the centroid of the opposite face

/-
Define the problem statement
-/

theorem medians_intersect_at_single_point (T : Tetrahedron) :
  ∃ O : Point, ∀ v : T.ABC.ABC.A | v : T.ABC.B | v : T.ABC.C | v : T.ABC.D,
  let G := (median T G v).end in
  (median T G v).measures_divisible_by O (3, 1) :=
sorry

end medians_intersect_at_single_point_l384_384905


namespace distance_traveled_by_center_of_ball_l384_384585

noncomputable def ball_diameter : ℝ := 6
noncomputable def ball_radius : ℝ := ball_diameter / 2
noncomputable def R1 : ℝ := 100
noncomputable def R2 : ℝ := 60
noncomputable def R3 : ℝ := 80
noncomputable def R4 : ℝ := 40

noncomputable def effective_radius_inner (R : ℝ) (r : ℝ) : ℝ := R - r
noncomputable def effective_radius_outer (R : ℝ) (r : ℝ) : ℝ := R + r

noncomputable def dist_travel_on_arc (R : ℝ) : ℝ := R * Real.pi

theorem distance_traveled_by_center_of_ball :
  dist_travel_on_arc (effective_radius_inner R1 ball_radius) +
  dist_travel_on_arc (effective_radius_outer R2 ball_radius) +
  dist_travel_on_arc (effective_radius_inner R3 ball_radius) +
  dist_travel_on_arc (effective_radius_outer R4 ball_radius) = 280 * Real.pi :=
by 
  -- Calculation steps can be filled in here but let's skip
  sorry

end distance_traveled_by_center_of_ball_l384_384585


namespace range_of_a_l384_384830

def tangent_perpendicular_to_y_axis (a : ℝ) : Prop :=
  ∃ (x : ℝ), x > 0 ∧ (3 * a * x^2 + 1 / x = 0)

theorem range_of_a : {a : ℝ | tangent_perpendicular_to_y_axis a} = {a : ℝ | a < 0} :=
by
  sorry

end range_of_a_l384_384830


namespace math_problem_l384_384298

variables (a b c : ℤ)

theorem math_problem (h1 : a - (b - 2 * c) = 19) (h2 : a - b - 2 * c = 7) : a - b = 13 := by
  sorry

end math_problem_l384_384298


namespace ratio_in_two_years_l384_384596

def son_age : ℕ := 22
def man_age : ℕ := son_age + 24

theorem ratio_in_two_years :
  (man_age + 2) / (son_age + 2) = 2 := 
sorry

end ratio_in_two_years_l384_384596


namespace sum_of_squares_l384_384157

theorem sum_of_squares (n : ℕ) (x : ℕ) (h1 : (x + 1)^3 - x^3 = n^2) (h2 : n > 0) : ∃ a b : ℕ, n = a^2 + b^2 :=
by
  sorry

end sum_of_squares_l384_384157


namespace exponent_problem_l384_384981

theorem exponent_problem : (3^3 * 3^(-5)) / (3^(-2) * 3^4) = 1 / 81 := 
by
  sorry

end exponent_problem_l384_384981


namespace ab_plus_c_l384_384377

open Int

theorem ab_plus_c (A B C : ℕ) (h_gcd : gcd (gcd A B) C = 1) (h_eq : A * log 500 5 + B * log 500 2 = log 500 500) :
  A + B + C = 6 := sorry

end ab_plus_c_l384_384377


namespace value_to_subtract_l384_384832

theorem value_to_subtract (N x : ℕ) 
  (h1 : (N - x) / 7 = 7) 
  (h2 : (N - 2) / 13 = 4) : x = 5 :=
by 
  sorry

end value_to_subtract_l384_384832


namespace parallel_lines_projections_suff_not_nec_l384_384977

noncomputable def lines_parallel_projections (a b m n : Type) [LinearOrder a] [LinearOrder b] [LinearOrder m] [LinearOrder n] : Prop :=
  -- Define the relationship between lines and their projections
  ∀ (α : Type) [AffineSpace α], 
    -- lines a and b are non-coincident lines on the plane α
    (∃ (a b : α), a ≠ b) ∧ 
    -- Projections of a and b on α are lines m and n, respectively, and are also non-coincident
    (∃ (m n : α), m ≠ n) →
    -- Sufficient condition
    (parallel a b → parallel m n) ∧ 
    -- Not necessary condition
    (parallel m n → ¬ parallel a b ∨ parallel a b)

theorem parallel_lines_projections_suff_not_nec (a b m n : Type) [LinearOrder a] [LinearOrder b] [LinearOrder m] [LinearOrder n] :
  lines_parallel_projections a b m n :=
by
  sorry

end parallel_lines_projections_suff_not_nec_l384_384977


namespace car_distance_calculation_l384_384254

noncomputable def total_distance (u a v t1 t2: ℝ) : ℝ :=
  let d1 := (u * t1) + (1 / 2) * a * t1^2
  let d2 := v * t2
  d1 + d2

theorem car_distance_calculation :
  total_distance 30 5 60 2 3 = 250 :=
by
  unfold total_distance
  -- next steps include simplifying the math, but we'll defer details to proof
  sorry

end car_distance_calculation_l384_384254


namespace segment_length_and_slope_max_min_value_MQ_max_min_slope_l384_384787

theorem segment_length_and_slope
    (a : ℝ)
    (Q : ℝ × ℝ := (-2, 3))
    (C : ℝ × ℝ → Prop := λ P, P.1^2 + P.2^2 - 4 * P.1 - 14 * P.2 + 45 = 0)
    (P : ℝ × ℝ := (a, a + 1))
    (hP : C P) :
    (Real.dist P Q = 2 * Real.sqrt 10) ∧ ((Q.2 - P.2) / (Q.1 - P.1) = 1 / 3) :=
by
  sorry

theorem max_min_value_MQ
    (M Q : ℝ × ℝ)
    (C : ℝ × ℝ → Prop := λ P, P.1^2 + P.2^2 - 4 * P.1 - 14 * P.2 + 45 = 0)
    (hM : C M)
    (hQ : Q = (-2, 3))
    (center : ℝ × ℝ := (2, 7))
    (radius : ℝ := 2 * Real.sqrt 2) :
    (Real.dist M Q ≤ Real.dist center Q + radius) ∧
    (Real.dist M Q ≥ Real.dist center Q - radius) ∧
    (Real.dist M Q = 6 * Real.sqrt 2 ∨ Real.dist M Q = 2 * Real.sqrt 2) :=
by
  sorry

theorem max_min_slope
    (m n : ℝ)
    (M Q : ℝ × ℝ := (m, n), (-2, 3))
    (C : ℝ × ℝ → Prop := λ P, P.1^2 + P.2^2 - 4 * P.1 - 14 * P.2 + 45 = 0)
    (hM : C (m, n)) :
    (2 - Real.sqrt 3 ≤ (n - 3) / (m + 2)) ∧
    ((n - 3) / (m + 2) ≤ 2 + Real.sqrt 3) :=
by
  sorry

end segment_length_and_slope_max_min_value_MQ_max_min_slope_l384_384787


namespace tan_alpha_eq_one_l384_384067

noncomputable def rho (theta : ℝ) : ℝ := sorry
noncomputable def parametric_line_x (t α : ℝ) : ℝ := 2 + t * Real.cos α
noncomputable def parametric_line_y (t α : ℝ) : ℝ := 3 + t * Real.sin α

theorem tan_alpha_eq_one (α : ℝ) (t : ℝ) (rho : ℝ → ℝ) :
  (∀ θ, rho θ * Real.sin θ^2 + 4 * Real.sin θ - rho θ = 0) →
  (parametric_line_x t α, parametric_line_y t α) = (0, 1) →
  Real.tan α = 1 :=
by
  simp [parametric_line_x, parametric_line_y, Real.tan]
  sorry

end tan_alpha_eq_one_l384_384067


namespace balance_balls_l384_384113

variables (R B O P : ℝ)

-- Conditions
axiom h1 : 4 * R = 8 * B
axiom h2 : 3 * O = 6 * B
axiom h3 : 8 * B = 6 * P

-- Proof problem
theorem balance_balls : 5 * R + 3 * O + 3 * P = 20 * B :=
by sorry

end balance_balls_l384_384113


namespace common_ratio_eq_neg2_l384_384189

-- Define the initial conditions for the geometric sequence and sums
variables {a1 q : ℝ}
def S1 := a1
def S2 := a1 * (1 + q)
def S3 := a1 * (1 + q + q^2)

-- The main theorem to be proved
theorem common_ratio_eq_neg2 (h : S3 + 3 * S2 = 0) (h₀ : a1 ≠ 0) : q = -2 :=
by
  sorry

end common_ratio_eq_neg2_l384_384189


namespace zeta_1_8_add_zeta_2_8_add_zeta_3_8_l384_384246

noncomputable def compute_s8 (s : ℕ → ℂ) : ℂ :=
  s 8

theorem zeta_1_8_add_zeta_2_8_add_zeta_3_8 {ζ : ℕ → ℂ} 
  (h1 : ζ 1 + ζ 2 + ζ 3 = 2)
  (h2 : ζ 1^2 + ζ 2^2 + ζ 3^2 = 6)
  (h3 : ζ 1^3 + ζ 2^3 + ζ 3^3 = 18)
  (rec : ∀ n, ζ (n + 3) = 2 * ζ (n + 2) + ζ (n + 1) - (4 / 3) * ζ n)
  (s0 : ζ 0 = 3)
  (s1 : ζ 1 = 2)
  (s2 : ζ 2 = 6)
  (s3 : ζ 3 = 18)
  : ζ 8 = compute_s8 ζ := 
sorry

end zeta_1_8_add_zeta_2_8_add_zeta_3_8_l384_384246


namespace solve_for_alpha_l384_384021

theorem solve_for_alpha : 
  ∀ (f : ℝ → ℝ) (α : ℝ), (f = λ x, 4 / (1 - x)) → (f α = 2) → (α = -1) :=
by
  intros f α hf hα
  sorry

end solve_for_alpha_l384_384021


namespace solve_logarithmic_equation_l384_384505

theorem solve_logarithmic_equation (x : ℝ) (h : log 8 x + 3 * log 2 (x^2) - log 4 x = 14) :
  x = 2^(12/5) :=
by
  sorry

end solve_logarithmic_equation_l384_384505


namespace nonexistence_tetrahedron_l384_384647

noncomputable theory

structure Tetrahedron (V : Type*) :=
(A B C D : V)
(length : V → V → ℝ)
(dihedral_angles_sum : ∀ {v : V}, v = B ∨ v = C → ℝ)

def valid_tetrahedron {V : Type*} (T : Tetrahedron V) : Prop :=
  T.length T.A T.B = T.length T.A T.C ∧
  T.length T.A T.B = T.length T.A T.D ∧
  T.length T.A T.B = T.length T.B T.C ∧
  T.dihedral_angles_sum T.B = 150 ∧ 
  T.dihedral_angles_sum T.C = 150

theorem nonexistence_tetrahedron {V : Type*} (T : Tetrahedron V) :
  ¬ (valid_tetrahedron T) :=
sorry

end nonexistence_tetrahedron_l384_384647


namespace num_terminating_decimals_l384_384684

theorem num_terminating_decimals (n : ℕ) (h1 : 1 ≤ n ∧ n ≤ 180) :
  (∃ k : ℕ, k = 20) := by
  have h2 : ∀ n, (1 ≤ n ∧ n ≤ 180) → (∃ k, k = (180 / 9)) := sorry
  have h3 : 180 / 9 = 20 := by norm_num
  use 20
  exact h3

end num_terminating_decimals_l384_384684


namespace Steiner_point_barycentric_l384_384657

noncomputable def barycentric_coords_Steiner (α β γ a b c : ℝ) : Prop :=
(β * γ + α * γ + α * β = 0) ∧
(a^2 * β * γ + b^2 * α * γ + c^2 * α * β = 0) →
(α / β = (a^2 - c^2) / (c^2 - b^2)) →
((1 / (b^2 - c^2)) : (1 / (c^2 - a^2)) : (1 / (a^2 - b^2)))

theorem Steiner_point_barycentric (a b c : ℝ) (h₁ h₂ h₃ : α β γ : ℝ) :
  barycentric_coords_Steiner α β γ a b c :=
sorry

end Steiner_point_barycentric_l384_384657


namespace maximum_angle_B_l384_384080

-- Let a, b, c be the sides of triangle ABC opposite to angles A, B, and C respectively.
variables {a b c : ℝ}

-- Let f be the given function f(x) = (1/3) * x^3 + b * x^2 + (a^2 + c^2 - a * c) * x + 1
def f (x : ℝ) : ℝ := (1 / 3) * x^3 + b * x^2 + (a^2 + c^2 - a * c) * x + 1

-- The condition that f(x) has no extreme points
def no_extreme_points : Prop := ∀ x : ℝ, (f' x)^2 - 4 * (a^2 + c^2 - a * c) * x^2 ≤ 0

-- We aim to prove that the maximum value of angle B is π/3
theorem maximum_angle_B (h : no_extreme_points) : 0 < B ∧ B ≤ π / 3 :=
sorry

end maximum_angle_B_l384_384080


namespace sum_x_square_over_1_minus_x_is_zero_l384_384887

theorem sum_x_square_over_1_minus_x_is_zero (x : Fin 150 → ℝ) 
  (h_sum : ∑ i, x i = 3) 
  (h_frac_sum : ∑ i, x i / (1 - x i) = 3) :
  ∑ i, (x i ^ 2) / (1 - x i) = 0 := 
by 
  sorry

end sum_x_square_over_1_minus_x_is_zero_l384_384887


namespace find_z_l384_384419

theorem find_z (a z : ℕ) (h₁ : a = 105) (h₂ : a^3 = 21 * 25 * 45 * z) : z = 49 :=
sorry

end find_z_l384_384419


namespace mindmaster_secret_codes_l384_384838

theorem mindmaster_secret_codes (colors slots : ℕ) (h_colors : colors = 6) (h_slots : slots = 5) : colors ^ slots = 7776 :=
by
  rw [h_colors, h_slots]
  norm_num
  sorry

end mindmaster_secret_codes_l384_384838


namespace paint_needed_for_320_statues_l384_384412

theorem paint_needed_for_320_statues 
  (paint_needed_for_original : ℕ = 2)
  (height_original : ℕ = 8)
  (height_small : ℕ = 2)
  (number_of_small_statues : ℕ = 320) : 
  ℕ :=
  let ratio_of_heights := height_small / height_original
  let surface_area_ratio := (ratio_of_heights : ℚ) ^ 2
  let paint_needed_for_small := (paint_needed_for_original : ℚ) * surface_area_ratio
  let total_paint_needed := (number_of_small_statues : ℚ) * paint_needed_for_small
  total_paint_needed.to_nat = 40 := by
  sorry

end paint_needed_for_320_statues_l384_384412


namespace delivery_newspapers_15_houses_l384_384601

-- State the problem using Lean 4 syntax

noncomputable def delivery_sequences (n : ℕ) : ℕ :=
  if h : n < 3 then 2^n
  else if n = 3 then 6
  else delivery_sequences (n-1) + delivery_sequences (n-2) + delivery_sequences (n-3)

theorem delivery_newspapers_15_houses :
  delivery_sequences 15 = 849 :=
sorry

end delivery_newspapers_15_houses_l384_384601


namespace symmetry_center_of_function_l384_384946

theorem symmetry_center_of_function : 
  ∃ (p : ℝ × ℝ), p = (-π / 16, 0) ∧ ∀ (x : ℝ), f(x) = 2 * sin (4 * x + π / 4) → 
  f(-π / 16 - (x + π / 16)) = f(x) :=
begin
  sorry
end

end symmetry_center_of_function_l384_384946


namespace Sally_bought_20_pokemon_cards_l384_384909

theorem Sally_bought_20_pokemon_cards
  (initial_cards : ℕ)
  (cards_from_dan : ℕ)
  (total_cards : ℕ)
  (bought_cards : ℕ)
  (h1 : initial_cards = 27)
  (h2 : cards_from_dan = 41)
  (h3 : total_cards = 88)
  (h4 : total_cards = initial_cards + cards_from_dan + bought_cards) :
  bought_cards = 20 := 
by
  sorry

end Sally_bought_20_pokemon_cards_l384_384909


namespace max_knights_at_table_l384_384115

variables (N : ℕ) (natives : List (Prop)) (knight : Prop -> Prop) (liar : Prop -> Prop)
variables (statement : list (Prop)) (mistakes : ℕ) (maxKnights : ℕ)

-- Condition: 2022 natives at the table
def totalNatives := 2022

-- Condition: Each native makes the statement: "Next to me sit a knight and a liar!"
def statement_truth (idx : ℕ) : Prop :=
  (knight (natives.get (idx - 1 % 2022)) ∧ liar (natives.get (idx + 1 % 2022))) ∨
  (liar (natives.get (idx - 1 % 2022)) ∧ knight (natives.get (idx + 1 % 2022)))

-- Condition: There are 3 knights who made a mistake
def knights_made_mistake := 3

-- Prove the maximum number of knights
theorem max_knights_at_table :
  ∀ (knights : ℕ),
  (knights ≤ maxKnights) ∧
  (totalNatives - knights <= 2022 - 1349) →
  knights ≤ 1349 :=
sorry

end max_knights_at_table_l384_384115


namespace find_m_l384_384136

theorem find_m (x y m : ℝ) (opp_sign: y = -x) 
  (h1 : 4 * x + 2 * y = 3 * m) 
  (h2 : 3 * x + y = m + 2) : 
  m = 1 :=
by 
  -- Placeholder for the steps to prove the theorem
  sorry

end find_m_l384_384136


namespace ellipse_parabola_intersection_l384_384786

theorem ellipse_parabola_intersection (a b k m : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c = Real.sqrt 2) (h4 : c^2 = a^2 - b^2)
    (h5 : (1 / 2) * 2 * a * 2 * b = 2 * Real.sqrt 3) (h6 : k ≠ 0) :
    (∃ (m: ℝ), (1 / 2) < m ∧ m < 2) :=
sorry

end ellipse_parabola_intersection_l384_384786


namespace light_bulbs_problem_l384_384447

-- Conditions
def grid : Type := fin 4 × fin 5
def bulbs (g : grid) : Prop := true

variable (initial_state : grid → bool)

-- Definitions for moves
def move_condition (move: grid → Prop) (initial_on : grid → bool) (final_on : grid → bool) : Prop :=
  ∃ l : grid → bool, 
    (∀ x, move x → ¬ l x) ∧  -- No bulbs on the "line"
    (∀ x, ¬ move x → l x = initial_on x) ∧  -- Bulbs on one side remain as they are 
    (∀ x, move x → final_on x = true) ∧  -- All bulbs on one side turn on
    (∃ x, move x ∧ ¬ initial_on x) -- At least one bulb must be initially off and turned on

def possible_to_turn_all_on (initial_state : grid → bool) : Prop :=
  ∃ (move1 move2 move3 move4 : grid → Prop), 
    move_condition move1 initial_state (final_state1) ∧
    move_condition move2 final_state1 (final_state2) ∧
    move_condition move3 final_state2 (final_state3) ∧
    move_condition move4 final_state3 (final_state4) ∧
    (∀ x, final_state4 x = true)

theorem light_bulbs_problem : possible_to_turn_all_on initial_state := 
by
  sorry

end light_bulbs_problem_l384_384447


namespace inverse_of_matrix_l384_384329

noncomputable def my_matrix := matrix ([[5, -3], [-2, 1]])

theorem inverse_of_matrix :
  ∃ (M_inv : matrix ℕ ℕ ℝ), (my_matrix.det ≠ 0) ∧ (my_matrix * M_inv = 1) → M_inv = matrix ([[ -1, -3 ], [-2, -5 ]]) :=
by
  sorry

end inverse_of_matrix_l384_384329


namespace trajectory_equation_lambda_range_l384_384026

noncomputable def hyperbola_eq (x y : ℝ) : Prop := (x^2 / 4) - y^2 = 1

noncomputable def ellipse_eq (x y : ℝ) : Prop := (x^2 / 4) + y^2 = 1

theorem trajectory_equation :
  (∀ (M : ℝ × ℝ), (∃ (t : ℝ), P (t) ∧ Q (-t) ∧ M = intersection A1P A2Q) → 
   ellipse_eq M.1 M.2)
:=
begin
  sorry
end

theorem lambda_range (λ : ℝ) :
  (∀ (E A B : ℝ × ℝ), E = (0, 2) ∧ on_trajectory A D ∧ on_trajectory B D
  → (vector EA = λ • vector EB) → λ ∈ Ioo (1 / 3) 3)
:=
begin
  sorry
end

end trajectory_equation_lambda_range_l384_384026


namespace angle_sum_l384_384071

theorem angle_sum (x : ℝ) (h1 : 2 * x + x = 90) : x = 30 := 
sorry

end angle_sum_l384_384071


namespace prime_divisibility_equiv_l384_384346

theorem prime_divisibility_equiv (p : ℕ) [Fact (Nat.Prime p)] :
  (∃ x0 : ℤ, p ∣ (x0^2 - x0 + 3)) ↔ (∃ y0 : ℤ, p ∣ (y0^2 - y0 + 25)) :=
sorry

end prime_divisibility_equiv_l384_384346


namespace line_point_t_l384_384652

theorem line_point_t (t : ℝ) : 
  (∃ t, (0, 3) = (0, 3) ∧ (-8, 0) = (-8, 0) ∧ (5 - 3) / t = 3 / 8) → (t = 16 / 3) :=
by
  sorry

end line_point_t_l384_384652


namespace roundness_of_8000000_l384_384633

def roundness (n : ℕ) : ℕ :=
  let factors := Nat.factors n
  let powers := List.foldr (λ x acc, if acc.keys.contains x then acc.insert x (acc.findD x 1 + 1) else acc.insert x 1) (Std.RBMap.empty Nat compare) factors
  powers.values.sum

theorem roundness_of_8000000 : roundness 8000000 = 15 := by
  sorry

end roundness_of_8000000_l384_384633


namespace points_collinear_sum_l384_384441

theorem points_collinear_sum (x y : ℝ) :
  ∃ k : ℝ, (x - 1 = 3 * k ∧ 1 = k * (y - 2) ∧ -1 = 2 * k) → 
  x + y = -1 / 2 :=
by
  sorry

end points_collinear_sum_l384_384441


namespace find_value_of_complex_fraction_l384_384005

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem find_value_of_complex_fraction :
  (1 - 2 * i) / (1 + i) = -1 / 2 - 3 / 2 * i := 
sorry

end find_value_of_complex_fraction_l384_384005


namespace terminating_decimal_fraction_count_l384_384685

theorem terminating_decimal_fraction_count :
  (finset.filter (λ n : ℕ, 1 ≤ n ∧ n ≤ 180 ∧ 90 ∣ n) (finset.range 181)).card = 20 :=
by sorry

end terminating_decimal_fraction_count_l384_384685


namespace lambda_range_l384_384781

theorem lambda_range (λ : ℝ) (a : ℕ → ℝ) (inc : ∀ n : ℕ, n > 0 → a n < a (n + 1)) :
  (∀ n : ℕ, n > 0 → a n = n^2 + λ * n) → λ > -3 :=
by
  intro h
  have h1 : a 1 = 1 + λ := by rw [h 1 (by norm_num)]
  have h2 : a 2 = 4 + 2 * λ := by rw [h 2 (by norm_num)]
  sorry

end lambda_range_l384_384781


namespace solution_pair_l384_384659

-- Define the conditions as functions
def equation1 (x y : ℝ) : Prop := y = (x + 1)^4
def equation2 (x y : ℝ) : Prop := x * y + y = Real.cos (π * x)

-- Formulate the proof problem
theorem solution_pair :
  ∀ (x y : ℝ), (equation1 x y ∧ equation2 x y) → (x = 0 ∧ y = 1) :=
by
  intros x y h
  sorry

end solution_pair_l384_384659


namespace combined_area_of_removed_triangles_l384_384610

theorem combined_area_of_removed_triangles (x r s : ℕ) (h1 : 2 * (r + s) + 2 * |r - s| = 32) (h2 : (r + s)^2 + (r - s)^2 = x^2) : 
  4 * (1/2) * r * s = x^2 / 2 :=
by 
  sorry

end combined_area_of_removed_triangles_l384_384610


namespace gcd_three_digit_palindromes_l384_384562

theorem gcd_three_digit_palindromes :
  let palindromes := { n | ∃ A B : ℤ, 0 ≤ A ∧ A ≤ 9 ∧ B = 0 ∨ B = 2 ∨ B = 4 ∨ B = 6 ∨ B = 8 ∧ n = 101 * A + 10 * B }
  ∃ d : ℤ, d > 0 ∧ d divides ∀ n ∈ palindromes,
  ∀ d', d' > 0 ∧ d' divides ∀ n ∈ palindromes → d' ≤ d
  ∧ d = 2 :=
by 
  sorry

end gcd_three_digit_palindromes_l384_384562


namespace count_valid_lines_l384_384817

open Set Finset

-- Define the set of points of the form (i, j, k) where 1 ≤ i, j, k ≤ 5
def points := { p : ℕ × ℕ × ℕ | p.1 ∈ Icc 1 5 ∧ p.2 ∈ Icc 1 5 ∧ p.3 ∈ Icc 1 5 }

-- Define the predicate to check if a line passes through exactly four distinct points in the set
def is_valid_line (p1 p2 p3 p4 : ℕ × ℕ × ℕ) : Prop :=
  p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧ p4 ∈ points ∧
  p1 ≠ p2 ∧ p2 ≠ p3 ∧ p3 ≠ p4 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p4 ∧
  collinear {p1, p2, p3, p4}

-- Define the collinearity condition
def collinear (s : Set (ℕ × ℕ × ℕ)) : Prop :=
  ∃ (u : ℕ × ℕ × ℕ) (v : ℕ × ℕ × ℕ),
  ∀ p ∈ s, ∃ k : ℤ, p = (u.1 + k * v.1, u.2 + k * v.2, u.3 + k * v.3)

theorem count_valid_lines : 
  ∃ (l : Finset (ℕ × ℕ × ℕ × ℕ × ℕ × ℕ × ℕ × ℕ × ℕ)), 
  (∀ a ∈ l, is_valid_line a.1 a.2 a.3 a.4) ∧ l.card = 140 :=
by sorry

end count_valid_lines_l384_384817


namespace terminating_decimals_l384_384705

theorem terminating_decimals (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 180) :
  (∃ k : ℕ, k = 20 ∧ ∀ m : ℕ, 1 ≤ m ∧ m ≤ 180 → (decimal_terminates (m / 180) ↔ m % 9 = 0)) :=
by {
  sorry
}

def decimal_terminates (x : ℚ) : Prop :=
  sorry

end terminating_decimals_l384_384705


namespace group_purchase_cheaper_l384_384238

-- Define the initial conditions
def initial_price : ℕ := 10
def bulk_price : ℕ := 7
def delivery_cost : ℕ := 100
def group_size : ℕ := 50

-- Define the costs for individual and group purchases
def individual_cost : ℕ := initial_price
def group_cost : ℕ := bulk_price + (delivery_cost / group_size)

-- Statement to prove: cost per participant in a group purchase is less than cost per participant in individual purchases
theorem group_purchase_cheaper : group_cost < individual_cost := by
  sorry

end group_purchase_cheaper_l384_384238


namespace general_term_proof_sum_b_seq_proof_l384_384367

section ArithmeticSequence
variable (a b : ℝ) (n : ℕ)
variable (a_seq : ℕ → ℝ) := λ n, a + (n-1) * b

-- Condition: Solution set of the inequality
variable (sol_set : Set ℝ)
hypothesis sol_eq : sol_set = {x | ax^2 - 3x + 2 > 0} = {x | x < 1 ∨ x > b}

def general_term_formula (a b : ℝ) : ℕ → ℝ := λ n, 2 * n - 1

noncomputable def b_seq (a_seq : ℕ → ℝ) (n : ℕ) : ℝ :=
  1 / (a_seq n * a_seq (n + 1))

noncomputable def sum_b_seq (b_seq : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum b_seq

theorem general_term_proof (h₁ : b = 2) (h₂ : a = 1) :
  general_term_formula a b = λ n, 2 * n - 1 :=
  sorry

theorem sum_b_seq_proof (a b : ℝ) (n : ℕ)
  (h₁ : a = 1) (h₂ : b = 2) :
  sum_b_seq (b_seq (general_term_formula a b)) n = n / (2 * n + 1) :=
  sorry
end ArithmeticSequence

end general_term_proof_sum_b_seq_proof_l384_384367


namespace hyperbola_eccentricity_l384_384874

-- Defining the hyperbola parameters and conditions
variables {a b c : ℝ} {P F₁ F₂ : ℝ × ℝ}
def hyperbola (x y : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = 1
def foci_distance (F₁ F₂ : ℝ × ℝ) : ℝ := dist F₁ F₂

-- Defining the given conditions from the problem
def conditions (P F₁ F₂ : ℝ × ℝ) : Prop :=
  let d := foci_distance F₁ F₂ in
  hyperbola P.1 P.2 ∧
  dist P F₂ = d ∧
  dist P F₁ = 4 * b ∧
  (let c := 2 * b - a in c^2 = a^2 + b^2)

-- Statement of the proof problem
theorem hyperbola_eccentricity (P F₁ F₂ : ℝ × ℝ) (a b : ℝ) (h : conditions P F₁ F₂) :
  let c := 2 * b - a in
  let e := c / a in
  e = 5 / 3 :=
sorry

end hyperbola_eccentricity_l384_384874


namespace probability_both_dice_same_color_l384_384227

-- Definitions according to the conditions
def num_sides_total := 30
def num_red_sides := 6
def num_green_sides := 8
def num_blue_sides := 10
def num_golden_sides := 6

-- Definition of the probability calculation for each color
def probability_same_color (n : ℕ) (total : ℕ) : ℚ := (n * n : ℚ) / (total * total : ℚ)

-- Combined probability of both dice showing the same color
def combined_probability :=
    probability_same_color num_red_sides num_sides_total + 
    probability_same_color num_green_sides num_sides_total + 
    probability_same_color num_blue_sides num_sides_total + 
    probability_same_color num_golden_sides num_sides_total

-- The final theorem statement
theorem probability_both_dice_same_color : combined_probability = 59 / 225 := 
    sorry -- Proof is not required

end probability_both_dice_same_color_l384_384227


namespace terminating_decimal_count_l384_384754

theorem terminating_decimal_count : 
  (Finset.filter (λ n, n % 3 = 0) (Finset.range 181)).card = 60 := by
  sorry

end terminating_decimal_count_l384_384754


namespace probability_prime_or_odd_l384_384914

-- Define the set of balls and their corresponding numbers
def balls : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}

-- Define the set of prime numbers among the balls
def primes : Finset ℕ := {2, 3, 5, 7}

-- Define the set of odd numbers among the balls
def odds : Finset ℕ := {1, 3, 5, 7}

-- Define the set of numbers that are either prime or odd
def primes_or_odds := primes ∪ odds

-- Calculate the probability as the ratio of the size of primes_or_odds set to the size of balls set
def probability := (primes_or_odds.card : ℚ) / balls.card

-- Statement that the probability is 5/7
theorem probability_prime_or_odd : probability = 5 / 7 := by
  sorry

end probability_prime_or_odd_l384_384914


namespace determine_x_l384_384834

variable (n p : ℝ)

-- Definitions based on conditions
def x (n : ℝ) : ℝ := 4 * n
def percentage_condition (n p : ℝ) : Prop := 2 * n + 3 = (p / 100) * 25

-- Statement to be proven
theorem determine_x (h : percentage_condition n p) : x n = 4 * n := by
  sorry

end determine_x_l384_384834


namespace curve_C_eq_max_area_quadrilateral_fixed_line_N_l384_384792

-- Definitions based on the conditions.
def O (x y : ℝ) : Prop := x^2 + y^2 = 16
def A : ℝ × ℝ := (6, 0)
def on_circle (B : ℝ × ℝ) : Prop := O B.1 B.2
def M (B : ℝ × ℝ) (M : ℝ × ℝ) : Prop := M.1 = (A.1 + B.1)/2 ∧ M.2 = B.2/2

-- Proof statements
theorem curve_C_eq : ∀ M, (∃ B, on_circle B ∧ M (B) M) → (M.1 - 3)^2 + M.2^2 = 4 :=
sorry

theorem max_area_quadrilateral : ∀ T l E F G H, 
  T = (2, 0) → intersects_curve_C l T E F → 
  perpendicular_line_through_T l T G H → 
  area_quadrilateral E F G H = 7 :=
sorry

theorem fixed_line_N : ∀ P Q E F N, 
  intersects_x_axis_C P Q → 
  intersects_PE_QF E F N P Q → 
  N.1 = -1 :=
sorry

end curve_C_eq_max_area_quadrilateral_fixed_line_N_l384_384792


namespace prob_B_win_4_1_prob_A_win_more_than_5_games_l384_384062

theorem prob_B_win_4_1 (p_win : ℚ) (p_B_win_4_1 : ℚ) (h_win : p_win = 1 / 2) :
  p_B_win_4_1 = 1 / 8 :=
by
  -- Define B winning 3 out of the first 4 games, and also winning the 5th game.
  let p_B_wins_3_of_4 := (nat.C(4, 3) * (p_win ^ 3) * (1 - p_win)) * p_win
  let p_B_wins_last_game := p_win
  have calc : p_B_win_4_1 = p_B_wins_3_of_4 * p_B_wins_last_game,
  sorry

theorem prob_A_win_more_than_5_games (p_win : ℚ) (p_A_win_more_5 : ℚ) (h_win : p_win = 1 / 2) :
  p_A_win_more_5 = 5 / 16 :=
by
  -- Define A winning 3 out of the first 5 games, and also winning the 6th game.
  let p_A_wins_4_2 := (nat.C(5, 3) * (p_win ^ 3) * (1 - p_win) ^ 2) * p_win
  -- Define A winning 3 out of the first 6 games, and also winning the 7th game.
  let p_A_wins_4_3 := (nat.C(6, 3) * (p_win ^ 3) * (1 - p_win) ^ 3) * p_win
  have calc : p_A_win_more_5 = p_A_wins_4_2 + p_A_wins_4_3,
  sorry

end prob_B_win_4_1_prob_A_win_more_than_5_games_l384_384062


namespace find_common_ratio_l384_384009

open Classical

noncomputable def geom_sum (a : ℕ → ℝ) (q : ℝ) : ℕ → ℝ :=
λ n, a 0 * (1 - q^(n + 1)) / (1 - q)

theorem find_common_ratio
  (a : ℕ → ℝ)
  (q : ℝ)
  (h : ∀ (k : ℕ), ∃ (lim_eq : ℝ), 
    ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |geom_sum a q n - geom_sum a q (k + 1) - lim_eq| < ε) :
  q = (Real.sqrt 5 - 1) / 2 :=
begin
  sorry
end

end find_common_ratio_l384_384009


namespace solve_equation_l384_384925

def equation_holds (x : ℝ) : Prop := 
  (1 / (x + 10)) + (1 / (x + 8)) = (1 / (x + 11)) + (1 / (x + 7))

theorem solve_equation : equation_holds (-9) :=
by
  sorry

end solve_equation_l384_384925


namespace terminating_decimal_fraction_count_l384_384692

theorem terminating_decimal_fraction_count :
  (finset.filter (λ n : ℕ, 1 ≤ n ∧ n ≤ 180 ∧ 90 ∣ n) (finset.range 181)).card = 20 :=
by sorry

end terminating_decimal_fraction_count_l384_384692


namespace angle_AKC_90_degrees_l384_384092

variables {A B C M N K : Type} [IsMidpoint M A C] [ReflectIn M B C N] [ParallelLineThrough N AC Meets AB K]

theorem angle_AKC_90_degrees 
  (h_isosceles_ABC : ∀ (A B C : Type), ∆ ABC is isosceles with AB = AC)
  (h_midpoint_M : M is the midpoint of AC)
  (h_reflect_M_N : N is the reflection of M in BC)
  (h_parallel_N_AC : line through N parallel to AC meets AB at K) :
  ∠ AKC = 90° :=
sorry

end angle_AKC_90_degrees_l384_384092


namespace identify_property_p_functions_l384_384784

def is_property_p (f : ℂ → ℝ) := ∀ (z1 z2 : ℂ) (λ : ℝ), 
  f (λ • z1 + (1 - λ) • z2) = λ * f z1 + (1 - λ) * f z2

def f1 (z : ℂ) : ℝ := z.re - z.im
def f2 (z : ℂ) : ℝ := z.re^2 - z.im
def f3 (z : ℂ) : ℝ := 2 * z.re + z.im

theorem identify_property_p_functions : 
  (is_property_p f1) ∧ ¬(is_property_p f2) ∧ (is_property_p f3) :=
by
  split
  · intros z1 z2 λ
    unfold f1
    simp
    sorry
  split
  · intros z1 z2 λ
    unfold f2
    simp
    sorry
  · intros z1 z2 λ
    unfold f3
    simp
    sorry

end identify_property_p_functions_l384_384784


namespace average_speed_of_motorcycle_trip_l384_384599

-- Define the conditions as hypotheses
def motorcycle_trip : Prop :=
  ∃ (d₁ d₂ s₁ s₂ total_distance total_time average_speed : ℝ),
    d₁ = 60 ∧
    s₁ = 20 ∧
    d₂ = 60 ∧
    s₂ = 30 ∧
    total_distance = d₁ + d₂ ∧
    total_time = d₁ / s₁ + d₂ / s₂ ∧
    average_speed = total_distance / total_time

-- The theorem to prove the average speed
theorem average_speed_of_motorcycle_trip : motorcycle_trip :=
  ∃ (d₁ d₂ s₁ s₂ total_distance total_time average_speed : ℝ),
    d₁ = 60 ∧
    s₁ = 20 ∧
    d₂ = 60 ∧
    s₂ = 30 ∧
    total_distance = d₁ + d₂ ∧
    total_time = d₁ / s₁ + d₂ / s₂ ∧
    average_speed = total_distance / total_time ∧
    average_speed = 24 :=
sorry

end average_speed_of_motorcycle_trip_l384_384599


namespace constant_term_is_sixth_term_l384_384658

theorem constant_term_is_sixth_term :
  let T := λ r : ℕ, binomial 10 r * (x^10 / x^(2 * r)) 
  in (∃ r : ℕ, T r = 0 ∧ r = 5) ↔ r = 5 → T (r + 1) = C :
  sorry

end constant_term_is_sixth_term_l384_384658


namespace complex_fraction_identity_l384_384044

theorem complex_fraction_identity (x y z a b c : ℝ)
  (h1 : x / a + y / b + z / c = 1)
  (h2 : a / x + b / y + c / z = 3) :
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 1 / 3 :=
by 
  sorry

end complex_fraction_identity_l384_384044


namespace books_from_first_shop_l384_384502

def price1 : ℕ := 1150
def price2 : ℕ := 920
def books2 : ℕ := 50
def avg_price : ℕ := 18
def total_price : ℕ := price1 + price2
def total_books (x : ℕ) : ℕ := x + books2

theorem books_from_first_shop (x : ℕ) (h : 2070 = 18 * total_books x) : x = 65 :=
by {
  have h_total_price : total_price = 2070,
  { simp [price1, price2, total_price] },
  
  rw [total_books] at h,
  simp at h,
  assumption
}

end books_from_first_shop_l384_384502


namespace direction_cosines_l384_384577

-- Definitions of the normal vectors based on the given conditions
def normal_vector1 : EuclideanSpace ℝ (Fin 3) := ![2, -3, -3]
def normal_vector2 : EuclideanSpace ℝ (Fin 3) := ![1, 2, 1]

-- Compute the direction vector as the cross product of the normals
def direction_vector : EuclideanSpace ℝ (Fin 3) := normal_vector1.cross_product normal_vector2

-- Magnitude of the direction vector
def magnitude_direction_vector : ℝ := Real.sqrt (direction_vector.1 ^ 2 + direction_vector.2 ^ 2 + direction_vector.3 ^ 2)

-- The direction cosines
def cos_alpha := direction_vector.1 / magnitude_direction_vector
def cos_beta := direction_vector.2 / magnitude_direction_vector
def cos_gamma := direction_vector.3 / magnitude_direction_vector

-- Theorem statement
theorem direction_cosines:
  cos_alpha = 3 / Real.sqrt 83 ∧ cos_beta = -5 / Real.sqrt 83 ∧ cos_gamma = 7 / Real.sqrt 83 := 
by 
  -- Proof steps would be here
  sorry

end direction_cosines_l384_384577


namespace games_this_year_proof_l384_384894

variable (missed_games : ℕ) (last_year_games : ℕ) (total_games : ℕ)
variable (this_year_games : ℕ)

-- Given conditions
def condition1 : missed_games = 41 := by sorry
def condition2 : last_year_games = 39 := by sorry
def condition3 : total_games = 54 := by sorry

-- Proof that Mike went to 15 games this year
theorem games_this_year_proof (h1 : missed_games = 41) (h2 : last_year_games = 39) (h3 : total_games = 54) :
    this_year_games = total_games - last_year_games :=
by
  rw [h3, h2]
  norm_num
  exact rfl

end games_this_year_proof_l384_384894


namespace first_year_after_2010_with_digit_sum_15_l384_384214

def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

theorem first_year_after_2010_with_digit_sum_15 : 
  ∃ y, y > 2010 ∧ sum_of_digits y = 15 ∧ ∀ z, z > 2010 ∧ sum_of_digits z = 15 → y ≤ z := 
begin
  use 2049,
  split,
  { linarith },
  split,
  { norm_num },
  { intros z hz,
    sorry },
end

end first_year_after_2010_with_digit_sum_15_l384_384214


namespace overall_gain_percentage_l384_384597

theorem overall_gain_percentage :
  let SP1 := 100
  let SP2 := 150
  let SP3 := 200
  let CP1 := SP1 / (1 + 0.20)
  let CP2 := SP2 / (1 + 0.15)
  let CP3 := SP3 / (1 - 0.05)
  let TCP := CP1 + CP2 + CP3
  let TSP := SP1 + SP2 + SP3
  let G := TSP - TCP
  let GP := (G / TCP) * 100
  GP = 6.06 := 
by {
  sorry
}

end overall_gain_percentage_l384_384597


namespace intersection_complement_l384_384029

open Set

variable (U A B : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5})
variable (hA : A = {1, 3, 4})
variable (hB : B = {4, 5})

theorem intersection_complement :
  A ∩ (U \ B) = {1, 3} :=
by
  rw [hU, hA, hB]
  ext
  simp
  sorry

end intersection_complement_l384_384029


namespace imaginary_part_of_z_eq_zero_l384_384167

theorem imaginary_part_of_z_eq_zero (z : ℂ) (h : z / (1 + 2 * complex.i) = 1 - 2 * complex.i) : z.im = 0 :=
sorry

end imaginary_part_of_z_eq_zero_l384_384167


namespace necessary_and_sufficient_condition_l384_384170

theorem necessary_and_sufficient_condition (a : ℝ) : (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≤ 0) ↔ (a ≥ 4) :=
by
  sorry

end necessary_and_sufficient_condition_l384_384170


namespace exists_color_removal_connected_l384_384544

noncomputable theory

open_locale classical

def K20_colored := simple_graph (fin 20) -- Define the complete graph K20

-- Define the complete graph with edges being one of five colors
axiom colored_edges : K20_colored → fin 5

theorem exists_color_removal_connected :
  ∃ c : fin 5, ∀ e, colored_edges e ≠ c → (K20_colored - {e}) .conn :=
sorry

end exists_color_removal_connected_l384_384544


namespace inverse_matrix_proof_l384_384333

variable (A : Matrix (Fin 2) (Fin 2) ℤ)
variable (B : Matrix (Fin 2) (Fin 2) ℤ)
variable (zeroMatrix : Matrix (Fin 2) (Fin 2) ℤ := ![(0, 0), (0, 0)])

-- Condition: The given matrices
def matrixA := ![(5, -3), (-2, 1)]
def matrixB := ![(-1, -3), (-2, -5)]

-- Property to prove: matrixB is the inverse of matrixA
theorem inverse_matrix_proof : 
  (∀ A : Matrix (Fin 2) (Fin 2) ℤ, A = matrixA) →
  (∀ B : Matrix (Fin 2) (Fin 2) ℤ, B = matrixB) →
  (B ⬝ A = 1) := 
  by sorry

end inverse_matrix_proof_l384_384333


namespace solve_equation_l384_384919

theorem solve_equation (x : ℝ) :
  (1 / (x + 10) + 1 / (x + 8) = 1 / (x + 11) + 1 / (x + 7)) → x = -9 :=
by 
  sorry

end solve_equation_l384_384919


namespace right_triangle_primes_l384_384436

-- Define what it means for a number to be a prime
def is_prime (n : ℕ) : Prop := ∃ m : ℕ, m > 1 ∧ m < n ∧ n % m = 0

-- State the problem
theorem right_triangle_primes
  (a b : ℕ)
  (ha : is_prime a)
  (hb : is_prime b)
  (a_gt_b : a > b)
  (a_plus_b : a + b = 90)
  (a_minus_b_prime : is_prime (a - b)) :
  b = 17 :=
sorry

end right_triangle_primes_l384_384436


namespace solve_for_x_l384_384415

theorem solve_for_x : ∀ (x : ℝ), 
  (x + 2 * x + 3 * x + 4 * x = 5) → (x = 1 / 2) :=
by 
  intros x H
  sorry

end solve_for_x_l384_384415


namespace sum_first_60_digits_of_fraction_l384_384223

theorem sum_first_60_digits_of_fraction (n : ℕ) (h : n = 60) :
  let decimal_expansion := "000891".cycle in
  (decimal_expansion.take n).foldr (λ c acc, acc + (c.to_nat - '0'.to_nat)) 0 = 180 :=
by sorry

end sum_first_60_digits_of_fraction_l384_384223


namespace terminating_decimals_count_l384_384733

theorem terminating_decimals_count : 
  let N := 180 in
  let count_terminating_decimals := (List.range' 1 N).filter (λ n, n % 3 = 0) in
  count_terminating_decimals.length = 60 :=
by
  let N := 180
  let count_terminating_decimals := (List.range' 1 N).filter (λ n, n % 3 = 0)
  sorry

end terminating_decimals_count_l384_384733


namespace shift_cos_to_sin_l384_384202

theorem shift_cos_to_sin (x : ℝ) : 
  cos (2 * x - π / 4) = sin (2 * (x - π / 8)) :=
by
  have h1 : sin (2 * x) = cos (2 * x - π / 2), from sorry
  have h2 : cos (2 * x - π / 4) = cos (2 * (x - π / 8) - π / 4), from sorry
  have h3 : cos (2 * (x - π / 8) - π / 4) = sin (2 * x), from sorry
  exact sorry

end shift_cos_to_sin_l384_384202


namespace find_a_l384_384794

def f (x : ℝ) : ℝ :=
if x > 1 then
  x^3
else
  -x^2 + 2*x

theorem find_a (a : ℝ) (h : f a = -5/4) : a = -1/2 := 
by sorry

end find_a_l384_384794


namespace inverse_matrix_proof_l384_384332

variable (A : Matrix (Fin 2) (Fin 2) ℤ)
variable (B : Matrix (Fin 2) (Fin 2) ℤ)
variable (zeroMatrix : Matrix (Fin 2) (Fin 2) ℤ := ![(0, 0), (0, 0)])

-- Condition: The given matrices
def matrixA := ![(5, -3), (-2, 1)]
def matrixB := ![(-1, -3), (-2, -5)]

-- Property to prove: matrixB is the inverse of matrixA
theorem inverse_matrix_proof : 
  (∀ A : Matrix (Fin 2) (Fin 2) ℤ, A = matrixA) →
  (∀ B : Matrix (Fin 2) (Fin 2) ℤ, B = matrixB) →
  (B ⬝ A = 1) := 
  by sorry

end inverse_matrix_proof_l384_384332


namespace line_through_Q_half_area_slope_intercept_sum_l384_384552

structure Point :=
  (x : ℝ)
  (y : ℝ)

def P := Point.mk 0 10
def Q := Point.mk 3 0
def R := Point.mk 9 0

def midpoint (A B : Point) : Point :=
  Point.mk ((A.x + B.x) / 2) ((A.y + B.y) / 2)

def slope (A B : Point) : ℝ :=
  (B.y - A.y) / (B.x - A.x)

def y_intercept (slope : ℝ) (A : Point) : ℝ :=
  A.y - slope * A.x

def line_equation (A B : Point) : (ℝ × ℝ) :=
  let m := slope A B
  let c := y_intercept m A
  (m, c)

def line_sum (m c : ℝ) : ℝ :=
  m + c

theorem line_through_Q_half_area_slope_intercept_sum :
  line_sum (line_equation Q (midpoint P R)).fst (line_equation Q (midpoint P R)).snd = -20/3 := by
  sorry

end line_through_Q_half_area_slope_intercept_sum_l384_384552


namespace relationship_abc_l384_384164

theorem relationship_abc {f : ℝ → ℝ} (h_diff : ∀ x : ℝ, differentiable_at ℝ f x)
  (h_symm : ∀ x : ℝ, f x = f (2 - x))
  (h_deriv : ∀ x : ℝ, (x - 1) * deriv f x < 0)
  (ha : a = f (2^(3/2)))
  (hb : b = f (real.log 2 / real.log 3))
  (hc : c = f (real.log 2 / real.log 0.5)) :
  c < a ∧ a < b := 
sorry

end relationship_abc_l384_384164


namespace num_divisors_90_l384_384040

theorem num_divisors_90 : (∀ (n : ℕ), n = 90 → (factors n).divisors.card = 12) :=
by {
  intro n,
  intro hn,
  sorry
}

end num_divisors_90_l384_384040


namespace sequence_length_is_3_l384_384529

def sequence (n : ℕ) : ℕ → ℕ
| 0     := n
| (k+1) := sequence k / 5

theorem sequence_length_is_3 (n : ℕ) (h_initial : n = 24300) (h_factorization : n = 2 * 3^5 * 5^2) :
  ∃ k : ℕ, k = 3 ∧ (∀ m : ℕ, m ≤ k → ∃ int_value : ℕ, sequence n m = int_value) :=
sorry

end sequence_length_is_3_l384_384529


namespace multiple_of_third_number_l384_384494

theorem multiple_of_third_number :
  ∃ k : ℤ, let first := 7
            let second := first + 2
            let third := second + 2
          in 8 * first = k * third + 5 + 2 * second ∧ k = 3 :=
by
  sorry

end multiple_of_third_number_l384_384494


namespace real_y_iff_x_conditions_l384_384048

theorem real_y_iff_x_conditions {x y : ℝ} :
  4 * y^2 + 6 * x * y + x + 8 = 0 →
  (y ∈ ℝ) ↔ (x ≤ -8 / 9 ∨ x ≥ 4) :=
by
  intro h
  sorry -- proof goes here

end real_y_iff_x_conditions_l384_384048


namespace terminating_decimal_integers_count_l384_384738

/-- For how many integer values of n between 1 and 180 inclusive does the decimal 
representation of n / 180 terminate? We need to calculate the number of integers n such 
that n includes at least 9 as a factor. The answer is 20. -/
theorem terminating_decimal_integers_count :
  { n : ℤ | 1 ≤ n ∧ n ≤ 180 ∧ 9 ∣ n }.to_finset.card = 20 := 
by sorry

end terminating_decimal_integers_count_l384_384738


namespace least_value_of_k_l384_384052

-- Define conditions
def is_integer (k : ℤ) : Prop := True

def exceeds_threshold (k : ℤ) : Prop :=
  0.0010101 * 10^k > 10

-- The main theorem stating the least value of k
theorem least_value_of_k : 
  ∃ k : ℤ, is_integer k ∧ exceeds_threshold k ∧ ∀ m : ℤ, (is_integer m ∧ exceeds_threshold m) → k ≤ m :=
begin
    use 5,
    split,
    { exact trivial },
    split,
    { dsimp [exceeds_threshold], linarith },
    { intros m hm, sorry }
end

end least_value_of_k_l384_384052


namespace number_of_valid_numbers_l384_384559

-- Define the digits
def digits : List ℕ := [0, 1, 2, 3, 4]

-- Check if a number is odd
def is_odd (n : ℕ) : Prop := n % 2 = 1

-- Check if a number is even
def is_even (n : ℕ) : Prop := n % 2 = 0

-- Define the property of a number having exactly one even number between two odd numbers
def has_one_even_sandwiched (l : List ℕ) : Prop :=
  ∃ (i : ℕ), i + 2 < l.length ∧ is_odd (l.nth i) ∧ is_even (l.nth (i + 1)) ∧ is_odd (l.nth (i + 2))

-- Define the condition where digits are used without repetition to form a five-digit number
def is_valid_five_digit_number (l : List ℕ) : Prop :=
  l.length = 5 ∧ l.nodup

-- Combine the main requirement with the conditions to form the final problem statement
theorem number_of_valid_numbers 
  (l : List (List ℕ)) (h1 : ∀ x ∈ l, is_valid_five_digit_number x)
  (h2 : ∀ x ∈ l, has_one_even_sandwiched x) : 
  l.length = 28 := 
sorry

end number_of_valid_numbers_l384_384559


namespace combinatorial_number_identity_l384_384939

theorem combinatorial_number_identity (n r : ℕ) (h1 : n > r) (h2 : r ≥ 1) :
  n.choose r = (n * (n - 1).choose (r - 1)) / r := 
begin
  sorry
end

end combinatorial_number_identity_l384_384939


namespace remainder_when_98_mul_102_divided_by_11_l384_384221

theorem remainder_when_98_mul_102_divided_by_11 :
  (98 * 102) % 11 = 1 :=
by
  sorry

end remainder_when_98_mul_102_divided_by_11_l384_384221


namespace weight_of_iron_pipe_l384_384996

theorem weight_of_iron_pipe (length : ℝ) (ext_diameter : ℝ) (thickness : ℝ) (density : ℝ) :
  length = 21 → ext_diameter = 8 → thickness = 1 → density = 8 →
  let ext_radius := ext_diameter / 2 in
  let int_radius := (ext_diameter - 2 * thickness) / 2 in
  let V_ext := Real.pi * (ext_radius ^ 2) * length in
  let V_int := Real.pi * (int_radius ^ 2) * length in
  let V_iron := V_ext - V_int in
  let weight := V_iron * density in
  weight ≈ 3694.68 :=
by
  intros
  simp at *
  sorry

end weight_of_iron_pipe_l384_384996


namespace terminating_decimals_l384_384702

theorem terminating_decimals (n : ℤ) (h : 1 ≤ n ∧ n ≤ 180) :
  (∃ k : ℤ, k = 20 ∧ ∀ n, 1 ≤ n ∧ n ≤ 180 → (∃ m, m * 180 = n * (2^2 * 5))) := by
  sorry

end terminating_decimals_l384_384702


namespace transform_eq_l384_384206

theorem transform_eq (x y : ℝ) (h : 5 * x - 6 * y = 4) : 
  y = (5 / 6) * x - (2 / 3) :=
  sorry

end transform_eq_l384_384206


namespace solve_equation_one_solve_equation_two_l384_384506

-- Define the first equation and its solutions
theorem solve_equation_one (x : ℝ) : x^2 - 6 * x = 1 ↔ (x = 3 + sqrt 10 ∨ x = 3 - sqrt 10) :=
by
  sorry

-- Define the second equation and its solutions
theorem solve_equation_two (x : ℝ) : (x - 3)^2 = (2 * x + 1)^2 ↔ (x = 2 / 3 ∨ x = -4) :=
by
  sorry

end solve_equation_one_solve_equation_two_l384_384506


namespace greatest_sum_l384_384563

theorem greatest_sum (x y : ℝ) (h1 : x^2 + y^2 = 100) (h2 : x * y = 40) : x + y = 6 * Real.sqrt 5 :=
sorry

end greatest_sum_l384_384563


namespace units_digit_of_result_l384_384191

def tens_plus_one (a b : ℕ) : Prop := a = b + 1

theorem units_digit_of_result (a b : ℕ) (h : tens_plus_one a b) :
  ((10 * a + b) - (10 * b + a)) % 10 = 9 :=
by
  -- Let's mark this part as incomplete using sorry.
  sorry

end units_digit_of_result_l384_384191


namespace var_decomposition_l384_384514

variables {Ω : Type*} {𝓕 : MeasurableSpace Ω} (ξ : Ω → ℝ) (𝒫 : Set (Set Ω)) 

-- Definition of Conditional Variance
def cond_var (ξ : Ω → ℝ) (𝒫 : Set (Set Ω)) :=
  E[λ ω, (ξ ω - E[λ ω', ξ ω' | 𝒫])^2 | 𝒫]

-- Definition of Variance
def variance (ξ : Ω → ℝ) : ℝ :=
  E[λ ω, (ξ ω - E[λ ω', ξ ω'])^2]

-- Goal: Decomposition of Variance
theorem var_decomposition :
  variance ξ = E[λ ω, cond_var ξ 𝒫] + variance (λ ω, E[λ ω', ξ ω' | 𝒫]) :=
sorry

end var_decomposition_l384_384514


namespace initial_people_per_column_l384_384837

theorem initial_people_per_column (P x : ℕ) (h1 : P = 16 * x) (h2 : P = 48 * 10) : x = 30 :=
by 
  sorry

end initial_people_per_column_l384_384837


namespace equation_of_tangent_circle_l384_384789

theorem equation_of_tangent_circle :
  (∃ (h k r : ℝ), 
    (∀ x y : ℝ, x - k + 1 = 0 ∧ x = 0 → h = 0 ∧ k = 1) ∧
    (∀ x y : ℝ, (x+y+1=0 → |1*k+1| / real.sqrt(2) = r ∧ r = √2)) ∧
    (∀ x y : ℝ, x^2 + (y - k)^2 = r^2) 
  → ∃ (x y : ℝ), x^2 + (y - 1)^2 = 2):=
by 
  sorry

end equation_of_tangent_circle_l384_384789


namespace mean_of_two_fractions_l384_384983

theorem mean_of_two_fractions :
  ( (2 : ℚ) / 3 + (4 : ℚ) / 9 ) / 2 = 5 / 9 :=
by
  sorry

end mean_of_two_fractions_l384_384983


namespace correct_option_c_l384_384224

-- Definitions for the problem context
noncomputable def qualification_rate : ℝ := 0.99
noncomputable def picking_probability := qualification_rate

-- The theorem statement that needs to be proven
theorem correct_option_c : picking_probability = 0.99 :=
sorry

end correct_option_c_l384_384224


namespace sample_size_calculation_l384_384591

theorem sample_size_calculation :
  let workshop_A := 120
  let workshop_B := 80
  let workshop_C := 60
  let sample_from_C := 3
  let sampling_fraction := sample_from_C / workshop_C
  let sample_A := workshop_A * sampling_fraction
  let sample_B := workshop_B * sampling_fraction
  let sample_C := workshop_C * sampling_fraction
  let n := sample_A + sample_B + sample_C
  n = 13 :=
by
  sorry

end sample_size_calculation_l384_384591


namespace minimum_n_for_obtuse_triangle_l384_384364

def α₀ : ℝ := 60 
def β₀ : ℝ := 59.999
def γ₀ : ℝ := 60.001

def α (n : ℕ) : ℝ := (-2)^n * (α₀ - 60) + 60
def β (n : ℕ) : ℝ := (-2)^n * (β₀ - 60) + 60
def γ (n : ℕ) : ℝ := (-2)^n * (γ₀ - 60) + 60

theorem minimum_n_for_obtuse_triangle : ∃ n : ℕ, β n > 90 ∧ ∀ m : ℕ, m < n → β m ≤ 90 :=
by sorry

end minimum_n_for_obtuse_triangle_l384_384364


namespace coefficient_x_squared_l384_384341

theorem coefficient_x_squared (a : ℝ) (x : ℝ) (h : x = 0.5) (eqn : a * x^2 + 9 * x - 5 = 0) : a = 2 :=
by
  sorry

end coefficient_x_squared_l384_384341


namespace terminating_decimals_count_l384_384728

theorem terminating_decimals_count : 
  let N := 180 in
  let count_terminating_decimals := (List.range' 1 N).filter (λ n, n % 3 = 0) in
  count_terminating_decimals.length = 60 :=
by
  let N := 180
  let count_terminating_decimals := (List.range' 1 N).filter (λ n, n % 3 = 0)
  sorry

end terminating_decimals_count_l384_384728


namespace solve_f100_eq_1_l384_384020

def f (x : ℝ) : ℝ := x / (x + 1)

def f_chain : ℕ → (ℝ → ℝ)
| 1     := f
| (n+1) := λ x, f (f_chain n x)

theorem solve_f100_eq_1 : (f_chain 100 x = 1) ↔ (x = -1/99) :=
by
  sorry

end solve_f100_eq_1_l384_384020


namespace rect_tiling_l384_384212

theorem rect_tiling (a b : ℕ) : ∃ (w h : ℕ), w = max 1 (2 * a) ∧ h = 2 * b ∧ (∃ f : ℕ → ℕ → (ℕ × ℕ), ∀ i j, (i < w ∧ j < h → f i j = (a, b))) := sorry

end rect_tiling_l384_384212


namespace terminating_decimals_count_l384_384724

theorem terminating_decimals_count :
  ∃ (count : ℕ), count = 60 ∧
  ∀ (n : ℕ), 1 ≤ n ∧ n ≤ 180 →
  (∃ m : ℕ, m * 180 = n * (2^2 * 5)) → 60 = count := sorry

end terminating_decimals_count_l384_384724


namespace second_row_number_is_3122_l384_384497

noncomputable def grid_value (A B C D E F G H I : ℕ) : Prop :=
  (A ≠ B) ∧ (B ≠ C) ∧ (C ≠ A) ∧ (D ≠ E) ∧ (E ≠ F) ∧ (F ≠ D) -- unique numbers in second row
  ∧ (A ∈ {1, 2, 3}) ∧ (B ∈ {1, 2, 3}) ∧ (C ∈ {1, 2, 3}) -- A, B, C initial assumptions
  ∧ (A = 1) ∧ (D = 3) ∧ (G = 2) -- Values for A, D, and G derived
  ∧ (F = 2) ∧ (H = 2) -- Values for F and H derived
  
theorem second_row_number_is_3122 :
  ∃ A B C D E F G H I : ℕ, grid_value A B C D E F G H I ∧ (E = 3 ∧ B = 1 ∧ F = 2 ∧ H = 2) := sorry

end second_row_number_is_3122_l384_384497


namespace cost_of_staying_23_days_l384_384421

-- Define the daily rates and number of days in a week
def first_week_rate := 18.00
def additional_week_rate := 12.00
def days_in_week := 7

-- Calculate cost for 7 days as the first week
def cost_first_week := days_in_week * first_week_rate

-- Calculate the remaining days after the first week
def remaining_days := 23 - days_in_week

-- Calculate costs for full additional weeks (14 days) and remaining days (2 days)
def full_additional_weeks := nat.floor (remaining_days / days_in_week)
def remaining_after_full_weeks := remaining_days % days_in_week

-- Cost calculations for additional weeks and remaining days
def cost_full_additional_weeks := full_additional_weeks * days_in_week * additional_week_rate
def cost_remaining_days := remaining_after_full_weeks * additional_week_rate

-- Total cost calculation
def total_cost := cost_first_week + cost_full_additional_weeks + cost_remaining_days

-- Final statement to prove
theorem cost_of_staying_23_days : total_cost = 318.00 := by
  sorry

end cost_of_staying_23_days_l384_384421


namespace find_m_l384_384007

theorem find_m (x1 x2 m : ℝ) (h1 : x1 + x2 = -3) (h2 : x1 * x2 = m) (h3 : 1 / x1 + 1 / x2 = 1) : m = -3 :=
by
  sorry

end find_m_l384_384007


namespace part1_solution_part2_solution_l384_384359

noncomputable def part1 (m : ℝ) (z : ℂ) : Prop :=
  z = (↑((m^2 + m - 6 : ℝ)) + ↑((m^2 + m - 2 : ℝ)) * complex.I) ∧
  (z - (↑(2 * m) : ℂ)).re = 0 ∧ m = 3

theorem part1_solution : ∀ (m : ℝ) (z : ℂ), part1 m z → m = 3 :=
by
  intros m z h,
  cases h with hz1 hz2,
  cases hz2 with h2 h3,
  exact h3

noncomputable def part2 (m : ℝ) : Prop :=
  (m^2 + m - 6 < 0) ∧ (m^2 + m - 2 > 0) ∧ 
  ((-3 < m ∧ m < -2) ∨ (1 < m ∧ m < 2))

theorem part2_solution : ∀ (m : ℝ), part2 m :=
by
  intros m h,
  cases h with h1 h2,
  cases h2 with h2 h3,
  exact h3

end part1_solution_part2_solution_l384_384359


namespace lines_intersect_and_find_point_l384_384851

theorem lines_intersect_and_find_point (n : ℝ)
  (h₁ : ∀ t : ℝ, ∃ (x y z : ℝ), x / 2 = t ∧ y / -3 = t ∧ z / n = t)
  (h₂ : ∀ t : ℝ, ∃ (x y z : ℝ), (x + 1) / 3 = t ∧ (y + 5) / 2 = t ∧ z / 1 = t) :
  n = 1 ∧ (∃ (x y z : ℝ), x = 2 ∧ y = -3 ∧ z = 1) :=
sorry

end lines_intersect_and_find_point_l384_384851


namespace total_seashells_l384_384131

-- Conditions
def sam_seashells : Nat := 18
def mary_seashells : Nat := 47

-- Theorem stating the question and the final answer
theorem total_seashells : sam_seashells + mary_seashells = 65 :=
by
  sorry

end total_seashells_l384_384131


namespace problem1_problem2_l384_384247

-- Problem 1
theorem problem1 {a b c : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (1 / a) + (1 / b) + (1 / c) ≥ (1 / (Real.sqrt (a * b))) + (1 / (Real.sqrt (b * c))) + (1 / (Real.sqrt (a * c))) :=
sorry

-- Problem 2
theorem problem2 {x y : ℝ} :
  Real.sin x + Real.sin y ≤ 1 + Real.sin x * Real.sin y :=
sorry

end problem1_problem2_l384_384247


namespace g_neg3_equals_neg7_l384_384802

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_inverse (f g : ℝ → ℝ) : Prop := ∀ y, g (f y) = y

-- Given conditions
def f (x : ℝ) : ℝ := if x >= 0 then log (x + 1) else -log (-x + 1)

axiom f_is_odd : is_odd f
axiom g_is_inverse : ∀ (g : ℝ → ℝ), is_inverse f g

-- Prove that g(-3) = -7
theorem g_neg3_equals_neg7 : ∃ g, is_inverse f g ∧ g (-3) = -7 :=
sorry

end g_neg3_equals_neg7_l384_384802


namespace side_opposite_larger_angle_longer_l384_384904

theorem side_opposite_larger_angle_longer (A B C : Point) (h : Triangle A B C) (h1 : angle C > angle B) : side A B > side A C := 
sorry

end side_opposite_larger_angle_longer_l384_384904


namespace max_knights_at_table_l384_384116

variables (N : ℕ) (natives : List (Prop)) (knight : Prop -> Prop) (liar : Prop -> Prop)
variables (statement : list (Prop)) (mistakes : ℕ) (maxKnights : ℕ)

-- Condition: 2022 natives at the table
def totalNatives := 2022

-- Condition: Each native makes the statement: "Next to me sit a knight and a liar!"
def statement_truth (idx : ℕ) : Prop :=
  (knight (natives.get (idx - 1 % 2022)) ∧ liar (natives.get (idx + 1 % 2022))) ∨
  (liar (natives.get (idx - 1 % 2022)) ∧ knight (natives.get (idx + 1 % 2022)))

-- Condition: There are 3 knights who made a mistake
def knights_made_mistake := 3

-- Prove the maximum number of knights
theorem max_knights_at_table :
  ∀ (knights : ℕ),
  (knights ≤ maxKnights) ∧
  (totalNatives - knights <= 2022 - 1349) →
  knights ≤ 1349 :=
sorry

end max_knights_at_table_l384_384116


namespace feet_of_perpendiculars_eq_l384_384614

noncomputable def triangle :=
  {A B C : Point}

noncomputable def centroid (t : triangle) : Point :=
sorry  -- Assume we have the centroid function definition

noncomputable def foot_of_perpendicular (P : Point) (line : Line) : Point :=
sorry  -- Assume the function to find foot of perpendicular is defined

theorem feet_of_perpendiculars_eq {t : triangle} (G : Point) 
  (hG : G = centroid t) (line : Line)
  (hline : G ∈ line)
  (X Y Z : Point) 
  (hX : X = foot_of_perpendicular t.A line)
  (hY : Y = foot_of_perpendicular t.C line)
  (hZ : Z = foot_of_perpendicular t.B line) :
  distance t.C Y = distance t.A X + distance t.B Z :=
sorry

end feet_of_perpendiculars_eq_l384_384614


namespace greatest_possible_percentage_of_airlines_l384_384839

/-- 
In a region, 50% of major airline companies equip their planes with wireless internet access. 
Among these companies, only 60% offer this service in all their planes whereas the remaining offer 
in some of their planes. On the other hand, 70% of major airlines offer passengers free on-board 
snacks. Out of these, 80% offer both salty and sweet snacks throughout their flights, while the 
rest provide only on long haul flights. 
-/
def greatest_possible_percentage : ℝ := 0.30

theorem greatest_possible_percentage_of_airlines :
  ∃ p : ℝ, 
  (let p := 0.50 * 0.60 in p = 0.30) →
  (let q := 0.70 * 0.80 in q = 0.56) →
  greatest_possible_percentage = 0.30 :=
by
  sorry

end greatest_possible_percentage_of_airlines_l384_384839


namespace most_circular_ellipse_l384_384385

theorem most_circular_ellipse : 
  ∀ α : ℝ, 0 < α ∧ α < (π / 2) → 
    (∃ x y : ℝ,
      (x^2 / tan α) + (y^2 / (tan α^2 + 1)) = 1 ∧
      (x^2 + y^2 / 2 = 1)) :=
begin
  sorry
end

end most_circular_ellipse_l384_384385


namespace nth_position_equation_l384_384111

theorem nth_position_equation (n : ℕ) (h : n > 0) : 9 * (n - 1) + n = 10 * n - 9 := by
  sorry

end nth_position_equation_l384_384111


namespace tan_shift_symmetric_l384_384949

theorem tan_shift_symmetric :
  let f (x : ℝ) := Real.tan (2 * x + Real.pi / 6)
  let g (x : ℝ) := f (x + Real.pi / 6)
  g (Real.pi / 4) = 0 ∧ ∀ x, g (Real.pi / 2 - x) = -g (Real.pi / 2 + x) :=
by
  sorry

end tan_shift_symmetric_l384_384949


namespace terminating_decimals_count_l384_384723

theorem terminating_decimals_count :
  ∃ (count : ℕ), count = 60 ∧
  ∀ (n : ℕ), 1 ≤ n ∧ n ≤ 180 →
  (∃ m : ℕ, m * 180 = n * (2^2 * 5)) → 60 = count := sorry

end terminating_decimals_count_l384_384723


namespace num_terminating_decimals_l384_384677

theorem num_terminating_decimals (n : ℕ) (h1 : 1 ≤ n ∧ n ≤ 180) :
  (∃ k : ℕ, k = 20) := by
  have h2 : ∀ n, (1 ≤ n ∧ n ≤ 180) → (∃ k, k = (180 / 9)) := sorry
  have h3 : 180 / 9 = 20 := by norm_num
  use 20
  exact h3

end num_terminating_decimals_l384_384677


namespace intersection_of_M_and_N_l384_384001

open Set

theorem intersection_of_M_and_N :
  let M := {-2, -1, 0, 1}
  let N := {x : ℤ | -3 ≤ x ∧ x < 0}
  M ∩ N = {-2, -1} := 
begin
  sorry
end

end intersection_of_M_and_N_l384_384001


namespace jordan_more_novels_than_maxime_l384_384313

def jordan_french_novels : ℕ := 130
def jordan_spanish_novels : ℕ := 20

def alexandre_french_novels : ℕ := jordan_french_novels / 10
def alexandre_spanish_novels : ℕ := 3 * jordan_spanish_novels

def camille_french_novels : ℕ := 2 * alexandre_french_novels
def camille_spanish_novels : ℕ := jordan_spanish_novels / 2

def total_french_novels : ℕ := jordan_french_novels + alexandre_french_novels + camille_french_novels

def maxime_french_novels : ℕ := total_french_novels / 2 - 5
def maxime_spanish_novels : ℕ := 2 * camille_spanish_novels

def jordan_total_novels : ℕ := jordan_french_novels + jordan_spanish_novels
def maxime_total_novels : ℕ := maxime_french_novels + maxime_spanish_novels

def novels_difference : ℕ := jordan_total_novels - maxime_total_novels

theorem jordan_more_novels_than_maxime : novels_difference = 51 :=
sorry

end jordan_more_novels_than_maxime_l384_384313


namespace f_of_7_eq_6_l384_384763

noncomputable def f : ℕ → ℕ
| x := if x ≥ 9 then x - 3 else f (f (x + 4))

theorem f_of_7_eq_6 : f 7 = 6 := 
by {
  sorry -- proof to be filled in
}

end f_of_7_eq_6_l384_384763


namespace last_two_digits_2007_pow_20077_l384_384168

theorem last_two_digits_2007_pow_20077 : (2007 ^ 20077) % 100 = 7 := 
by sorry

end last_two_digits_2007_pow_20077_l384_384168


namespace ways_to_make_half_dollar_l384_384043

def nickel_block := 1 -- one nickel fills one 5-cent block
def dime_block := 2 -- one dime fills two 5-cent blocks
def quarter_block := 5 -- one quarter fills five 5-cent blocks

-- Total number of blocks to fill a half-dollar
def total_blocks := 10

-- count_ways counts the number of ways to fill total_blocks using coins defined above
def count_ways (n d q : ℕ) : ℕ :=
  let nb := n * nickel_block
  let db := d * dime_block
  let qb := q * quarter_block
  if qb + nb + db = total_blocks then 1 else 0

noncomputable def total_ways : ℕ :=
  -- Possible ways with one quarter (fills 5 blocks)
  let ways_with_one_quarter := ∑ n in Finset.range 6, count_ways n (10 - n * nickel_block) 1 in 
  -- Possible ways with no quarters (fills 0 blocks)
  let ways_with_no_quarters := ∑ n in Finset.range 11, count_ways n (10 - n * nickel_block) 0 in
  -- Summing possible ways from each case
  ways_with_one_quarter + ways_with_no_quarters

-- The proof that the total number of distinct ways is 17
theorem ways_to_make_half_dollar : total_ways = 17 := by
  -- Skip the proof
  sorry

end ways_to_make_half_dollar_l384_384043


namespace top_square_after_folds_l384_384492

def initial_grid : ℕ → ℕ
| 1 := 1 | 2 := 2 | 3 := 3
| 4 := 4 | 5 := 5 | 6 := 6
| 7 := 7 | 8 := 8 | 9 := 9

def fold_right_third (grid : ℕ → ℕ) : ℕ → ℕ
| 1 := grid 1 | 2 := grid 3 | 3 := grid 2
| 4 := grid 4 | 5 := grid 6 | 6 := grid 5
| 7 := grid 7 | 8 := grid 9 | 9 := grid 8

def fold_left_third (grid : ℕ → ℕ) : ℕ → ℕ
| 1 := grid 1 | 4 := grid 4 | 7 := grid 7
| 2 := grid 2 | 5 := grid 5 | 8 := grid 8
| 3 := grid 3 | 6 := grid 6 | 9 := grid 9

def fold_bottom_third (grid : ℕ → ℕ) : ℕ → ℕ
| 7 := grid 7 | 4 := grid 4 | 1 := grid 1
| 8 := grid 8 | 5 := grid 5 | 2 := grid 2
| 9 := grid 9 | 6 := grid 6 | 3 := grid 3

theorem top_square_after_folds :
  let initial_grid := initial_grid in
  let grid_after_fold1 := fold_right_third initial_grid in
  let grid_after_fold2 := fold_left_third grid_after_fold1 in
  let grid_after_fold3 := fold_bottom_third grid_after_fold2 in
  grid_after_fold3 1 = 7 :=
by
  let initial_grid := initial_grid
  let grid_after_fold1 := fold_right_third initial_grid
  let grid_after_fold2 := fold_left_third grid_after_fold1
  let grid_after_fold3 := fold_bottom_third grid_after_fold2
  sorry

end top_square_after_folds_l384_384492


namespace count_special_numbers_l384_384818

theorem count_special_numbers : 
  let numbers := {n : ℕ | 300 ≤ n ∧ n ≤ 799} in
  let is_special n := 
    let d1 := n / 100 in 
    let d2 := (n % 100) / 10 in 
    let d3 := n % 10 in 
    (d1 ≠ d2 ∧ d2 ≠ d3 ∧ d1 ≠ d3 ∧ 
     ((d1 < d2 ∧ d2 < d3) ∨ (d1 > d2 ∧ d2 > d3))) ∨
    ((d1 = d2 ∧ d2 ≠ d3) ∨ (d1 ≠ d2 ∧ d2 = d3) ∨ (d1 = d3 ∧ d2 ≠ d3)) in
  ∑ n in numbers, if is_special n then 1 else 0 = 50 :=
begin
  sorry
end

end count_special_numbers_l384_384818


namespace original_mean_is_200_l384_384954

theorem original_mean_is_200
  (n : ℕ) (decrement : ℕ) (updated_mean : ℕ) (original_mean : ℕ)
  (h1 : n = 50) (h2 : decrement = 9) (h3 : updated_mean = 191) :
  original_mean = 200 :=
by
  -- Variables Declaration
  let total_decrement := n * decrement
  let sum_updated_observations := n * updated_mean
  let sum_original_observations := sum_updated_observations + total_decrement
  let original_mean_check := sum_original_observations / n
  
  -- Prove the statement
  have h4 : total_decrement = 450 := by sorry
  have h5 : sum_updated_observations = 9550 := by sorry
  have h6 : sum_original_observations = 10000 := by sorry
  have h7 : original_mean_check = 200 := by sorry
  exact h7

end original_mean_is_200_l384_384954


namespace hexagon_octagon_areas_equal_l384_384627

noncomputable def area_between_circles_hexagon (s : ℝ) : ℝ :=
  let A1 := Real.cot (Real.pi / 6) -- Apothem of the hexagon
  let R1 := Real.csc (Real.pi / 6) -- Radius of the circumscribed circle around the hexagon
  π * (R1^2 - A1^2)

noncomputable def area_between_circles_octagon (s : ℝ) : ℝ :=
  let A2 := Real.cot (Real.pi / 8) -- Apothem of the octagon
  let R2 := Real.csc (Real.pi / 8) -- Radius of the circumscribed circle around the octagon
  π * (R2^2 - A2^2)

theorem hexagon_octagon_areas_equal :
  let s := 2
  area_between_circles_hexagon s = area_between_circles_octagon s :=
sorry

end hexagon_octagon_areas_equal_l384_384627


namespace sin_alpha_valid_m_l384_384368

-- Problem 1: Prove that sin α = 2√5/5 given cos β = 3/5 and β = π - 2α
theorem sin_alpha (α β : Real) (hβ : β = π - 2 * α) (hcosβ : cos β = 3/5) (hα_pos : 0 < α) (hα_lt_pi_div_2 : α < π / 2) : sin α = 2 * √5 / 5 :=
sorry

-- Problem 2: Prove that 5π/12 ≤ m ≤ 5π/6 given the ranges of f(x) = tan x and g(x) = 2sin(2x-π/3) are the same on given intervals
theorem valid_m (α m : Real) (hα_pos : 0 < α) (hα_lt_pi_div_2 : α < π / 2) (hsinα : sin α = 2 * √5 / 5) (hrange : Set.range (fun x => tan x) (Icc (-π/3) α) = Set.range (fun x => 2 * sin (2*x - π/3)) (Icc 0 m)) : 5 * π / 12 ≤ m ∧ m ≤ 5 * π / 6 :=
sorry

end sin_alpha_valid_m_l384_384368


namespace min_value_of_modulus_l384_384474

noncomputable def smallest_possible_value (z : ℂ) (hz : |z - 7| + |z - 6 * complex.I| = 15) : ℝ :=
  |z|

theorem min_value_of_modulus : 
  ∃ z : ℂ, (|z - 7| + |z - 6 * complex.I| = 15) ∧ |z| = 14 / 5 :=
sorry

end min_value_of_modulus_l384_384474


namespace factor_t_squared_minus_81_l384_384321

theorem factor_t_squared_minus_81 (t : ℝ) : t^2 - 81 = (t - 9) * (t + 9) :=
by
  sorry

end factor_t_squared_minus_81_l384_384321


namespace terminating_decimal_count_l384_384746

theorem terminating_decimal_count : 
  (Finset.filter (λ n, n % 3 = 0) (Finset.range 181)).card = 60 := by
  sorry

end terminating_decimal_count_l384_384746


namespace min_value_of_xy_ratio_l384_384143

theorem min_value_of_xy_ratio :
  ∃ t : ℝ,
    (t = 2 ∨
    t = ((-1 + Real.sqrt 217) / 12) ∨
    t = ((-1 - Real.sqrt 217) / 12)) ∧
    min (min 2 ((-1 + Real.sqrt 217) / 12)) ((-1 - Real.sqrt 217) / 12) = -1.31 :=
sorry

end min_value_of_xy_ratio_l384_384143


namespace least_y_value_l384_384217

theorem least_y_value (y : ℝ) : 2 * y ^ 2 + 7 * y + 3 = 5 → y ≥ -2 :=
by
  intro h
  sorry

end least_y_value_l384_384217


namespace count_number_of_lines_l384_384434

theorem count_number_of_lines (P : Finset (Fin 7 → ℝ × ℝ))
  (hP : P.card = 7)
  (h_collinear : ∃ l : Line (ℝ × ℝ), {p ∈ P | p ∈ l}.card = 5)
  (h_no_three_collinear : ∀ l : Line (ℝ × ℝ), {p ∈ P | p ∈ l}.card ≤ 3) :
  ∃ lines : Finset (Line (ℝ × ℝ)), lines.card = 12 :=
sorry

end count_number_of_lines_l384_384434


namespace terminating_decimals_l384_384709

theorem terminating_decimals (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 180) :
  (∃ k : ℕ, k = 20 ∧ ∀ m : ℕ, 1 ≤ m ∧ m ≤ 180 → (decimal_terminates (m / 180) ↔ m % 9 = 0)) :=
by {
  sorry
}

def decimal_terminates (x : ℚ) : Prop :=
  sorry

end terminating_decimals_l384_384709


namespace part1_part2_l384_384465

open Set

def A : Set ℝ := {x | x^2 - 8 * x + 15 = 0}
def B (a : ℝ) : Set ℝ := {x | a * x = 1}

theorem part1 : B (1/5) ⊆ A ∧ ¬ A ⊆ B (1/5) := by
  sorry
  
theorem part2 (a : ℝ) : (B a ⊆ A) ↔ a ∈ ({0, 1/3, 1/5} : Set ℝ) := by
  sorry

end part1_part2_l384_384465


namespace cubic_inequality_l384_384929

theorem cubic_inequality (x : ℝ) : x^3 - 4*x^2 + 4*x < 0 ↔ x < 0 :=
by
  sorry

end cubic_inequality_l384_384929


namespace circle_tangent_standard_eq_l384_384426

theorem circle_tangent_standard_eq (x y : ℝ) (a : ℝ) : 
  (radius : ℝ), 
  (center_x : ℝ), 
  (center_y : ℝ) : 
  radius = 1 ∧ center_y = 1 ∧ center_x > 0 ∧ 
  (∃ (a : ℝ), 4 * center_x - 3 * center_y = a * sqrt (4^2 + (-3)^2) ∧ a = radius) → 
  (radius = 1) →
  ((x - 2)^2 + (y - 1)^2 = 1) := 
by 
  sorry

end circle_tangent_standard_eq_l384_384426


namespace solve_trig_problem_l384_384008

noncomputable def verify_results (α : ℝ) (x : ℝ) : Prop :=
  α ∈ set.Icc (π / 2) π ∧ -- α is in the second quadrant
  (x < 0) ∧
  (∃ P : ℝ × ℝ, P.1 = x ∧ P.2 = 4) ∧
  (cos α = x / 5) →
  (x = -3) ∧ (tan α = -4 / 3)

-- Now we can state the theorem to verify the results
theorem solve_trig_problem (α : ℝ) (x : ℝ) : verify_results α x :=
  by
    sorry

end solve_trig_problem_l384_384008


namespace inverse_of_matrix_l384_384327

noncomputable def my_matrix := matrix ([[5, -3], [-2, 1]])

theorem inverse_of_matrix :
  ∃ (M_inv : matrix ℕ ℕ ℝ), (my_matrix.det ≠ 0) ∧ (my_matrix * M_inv = 1) → M_inv = matrix ([[ -1, -3 ], [-2, -5 ]]) :=
by
  sorry

end inverse_of_matrix_l384_384327


namespace solution_set_l384_384295

-- Define the function f as specified by its properties.
def f (x: ℝ) : ℝ := if x ∈ set.Icc (-2) 2 then sqrt(1 - x^2 / 4) else 0

lemma f_odd (x : ℝ) : f (-x) = -f x :=
begin
  -- proof of odd function property
  sorry -- will be proved
end

-- Constraints based on ellipse equation.
lemma f_ellipse (x : ℝ) (hx: x ∈ set.Icc (-2) 2) : (x^2 / 4) + (f x)^2 = 1 :=
begin
  -- proof for ellipse constraint
  sorry -- will be proved
end

-- Statement to prove: solution set of inequality
theorem solution_set (x : ℝ) : 
  (-sqrt 2 < x ∧ x < 0) ∨ (sqrt 2 < x ∧ x ≤ 2) ↔ (f x < f (-x) + x) :=
by
  sorry -- will be proved

end solution_set_l384_384295


namespace chloe_paid_per_dozen_l384_384637

-- Definitions based on conditions
def half_dozen_sale_price : ℕ := 30
def profit : ℕ := 500
def dozens_sold : ℕ := 50
def full_dozen_sale_price := 2 * half_dozen_sale_price
def total_revenue := dozens_sold * full_dozen_sale_price
def total_cost := total_revenue - profit

-- Proof problem
theorem chloe_paid_per_dozen : (total_cost / dozens_sold) = 50 :=
by
  sorry

end chloe_paid_per_dozen_l384_384637


namespace gcd_polynomial_l384_384782

open Int

theorem gcd_polynomial {b : ℤ} (h : ∃ (k : ℤ), b = 17 * k ∧ Odd k) :
  gcd (3 * b^2 + 65 * b + 143) (5 * b + 22) = 33 :=
by
  sorry

end gcd_polynomial_l384_384782


namespace john_bought_3_reels_l384_384864

theorem john_bought_3_reels (reel_length section_length : ℕ) (n_sections : ℕ)
  (h1 : reel_length = 100) (h2 : section_length = 10) (h3 : n_sections = 30) :
  n_sections * section_length / reel_length = 3 :=
by
  sorry

end john_bought_3_reels_l384_384864


namespace part_a_part_b_part_c_l384_384392

def f (x : ℝ) : ℝ := cos (x + π / 3) * cos x - 1 / 4

theorem part_a : f (π / 3) = -1 / 2 := sorry

theorem part_b (k : ℤ) (x : ℝ) :
  k * π - 2 * π / 3 ≤ x ∧ x ≤ k * π - π / 6 → 
  ∃ C, ∀ x1 x2, C (f x1) (f x2) := sorry

theorem part_c (θ x : ℝ) (Hθ : 0 ≤ θ ∧ θ ≤ π / 2) :
  (∀ x, f(x + θ) = -f(x + θ)) → 
  ∃ range, range = (λ x, (f (x + π / 12))^2) '' (Ici (0 : ℝ)) := sorry

end part_a_part_b_part_c_l384_384392


namespace prime_not_fourth_power_sub_4_l384_384871

open Nat

theorem prime_not_fourth_power_sub_4 {p : ℕ} (hp_prime : Prime p) (hp_gt_5 : p > 5) :
  ¬ ∃ (q : ℕ), Prime q ∧ p - 4 = q ^ 4 := by
  sorry

end prime_not_fourth_power_sub_4_l384_384871


namespace locus_centroid_ZAB_is_origin_l384_384970

-- Define the conditions based on the problem statement

-- Positions after time 't' for all three cockroaches
variable (r : ℝ) (v : ℝ) (t : ℝ)
#check Real -- Check if Real type is imported correctly

def position_A (r v t : ℝ) := (r * Real.cos (v * t / r), r * Real.sin (v * t / r))
def position_B (r v t : ℝ) := (r * Real.cos (2 * v * t / r), r * Real.sin (2 * v * t / r))
def position_C (r v t : ℝ) := (r * Real.cos (3 * v * t / r), r * Real.sin (3 * v * t / r))

-- Points X and Y defined in terms of position C
def point_X (r v t : ℝ) :=
  ((2 * r + r * Real.cos (3 * v * t / r)) / 3, (r * Real.sin (3 * v * t / r)) / 3)

def point_Y (r v t : ℝ) :=
  ((r + 2 * r * Real.cos (3 * v * t / r)) / 3, (2 * r * Real.sin (3 * v * t / r)) / 3)

-- Define the centroid of triangle ZAB
def centroid (A B Z : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1 + Z.1) / 3, (A.2 + B.2 + Z.2) / 3)

-- Position a point (0,0)
def origin : ℝ × ℝ := (0, 0)

-- Proof statement asserting that the centroid of the triangle ZAB is always at (0,0)
theorem locus_centroid_ZAB_is_origin : 
  ∀ (r v t : ℝ),
    let A := position_A r v t in
    let B := position_B r v t in
    let Z := (0,0) in -- For simplicity, assuming Z = (0,0), based on symmetry 
    centroid A B Z = origin :=
by
  intros
  let A := position_A r v t
  let B := position_B r v t
  let Z := (0, 0)
  -- asserting that centroid should be origin
  sorry

end locus_centroid_ZAB_is_origin_l384_384970


namespace quadratic_has_two_distinct_real_roots_l384_384184

noncomputable def discriminant (a b c : ℝ) : ℝ := b * b - 4 * a * c

theorem quadratic_has_two_distinct_real_roots :
  (∀ x : ℝ, (x + 1) * (x - 1) = 2 * x + 3) → discriminant 1 (-2) (-4) > 0 :=
by
  intro h
  -- conditions from the problem
  let a := 1
  let b := -2
  let c := -4
  -- use the discriminant function directly with the values
  have delta := discriminant a b c
  show delta > 0
  sorry

end quadratic_has_two_distinct_real_roots_l384_384184


namespace division_631938_by_625_l384_384400

theorem division_631938_by_625 :
  (631938 : ℚ) / 625 = 1011.1008 :=
by
  -- Add a placeholder proof. We do not provide the solution steps.
  sorry

end division_631938_by_625_l384_384400


namespace largest_prime_divisor_l384_384275

-- Let n be a positive integer
def is_positive_integer (n : ℕ) : Prop :=
  n > 0

-- Define that n equals the sum of the squares of its four smallest positive divisors
def is_sum_of_squares_of_smallest_divisors (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), a = 2 ∧ b = 5 ∧ c = 10 ∧ n = 1 + a^2 + b^2 + c^2

-- Prove that the largest prime divisor of n is 13
theorem largest_prime_divisor (n : ℕ) (h1 : is_positive_integer n) (h2 : is_sum_of_squares_of_smallest_divisors n) :
  ∃ p : ℕ, Prime p ∧ p ∣ n ∧ ∀ q : ℕ, Prime q ∧ q ∣ n → q ≤ p ∧ p = 13 :=
by
  sorry

end largest_prime_divisor_l384_384275


namespace graph_symmetric_about_point_l384_384947

noncomputable def tan_shifted_symmetric : Prop :=
  ∀ x : ℝ, tan (2 * (x + π / 6) + π / 6) = g(x)

theorem graph_symmetric_about_point :
  ∀ x : ℝ, tan (2 * x + π / 2) = g(x) → g(π / 4) = 0 :=
sorry

end graph_symmetric_about_point_l384_384947


namespace functional_equations_lemma_l384_384306

open Real

variable {f g : ℝ → ℝ}

theorem functional_equations_lemma
  (h_nonconstant : ∃ x, f x ≠ 0 ∨ g x ≠ 0)
  (h_diff_f : Differentiable ℝ f)
  (h_diff_g : Differentiable ℝ g)
  (h_f1 : ∀ x y, f (x + y) = f x * f y - g x * g y)
  (h_g1 : ∀ x y, g (x + y) = f x * g y + g x * f y)
  (h_f_prime_zero : deriv f 0 = 0) :
    ∀ x, (f x)^2 + (g x)^2 = 1 := 
begin
  sorry
end

end functional_equations_lemma_l384_384306


namespace positive_integer_solution_count_l384_384401

theorem positive_integer_solution_count : 
  {x : ℤ // x > 0 ∧ 15 < -3 * x + 21}.card = 1 :=
by
  sorry

end positive_integer_solution_count_l384_384401


namespace no_single_beautiful_cell_l384_384440

-- Definition of the board
def board_size : ℕ := 100

-- Definition of a cell being beautiful
def is_beautiful (adj_chips: ℕ) : Prop :=
  adj_chips % 2 = 0

-- Main theorem statement: it's impossible to have exactly one beautiful cell
theorem no_single_beautiful_cell :
  ¬ ∃ (beautiful_count : Fin board_size.succ → Fin board_size.succ → ℕ),
    (∀ i j, beautiful_count i j = (∑ neighbour in (adjacent_cells i j), board_cells neighbour) % 2) ∧
    (∃! i j, beautiful_count i j = 0) :=
sorry

end no_single_beautiful_cell_l384_384440


namespace math_proof_problem_l384_384019

noncomputable def question1 (k : ℝ) : Prop :=
  let C := circle (0, 4) 2 in
  let l  := line (0, 0) k in
  ¬(discriminant (quad_eq (k*x) - ⟨4, [1+k^2]⟩ + 12) > 0 --> (k ∈ (-∞, -√(3)) ∪ (√(3), ∞)))

noncomputable def question2 (G : point) : Prop := 
  let C := circle (0, 4) 2 in
  let T := circle (0, 2) 2 in
  let diameter := central_angle (G ∈ T) 2π /3 in
  C ∧ T ∧ (G.trajectory C) = (G.trajectory T) ∧ (G.length T) = (4π/3)

noncomputable def question3 (m n : ℝ) : Prop :=
  let OQ := |(m, n)| 
  let OM := |(x1, k*x1)| 
  let ON := |(x2, k*x2)| 
  (2 / |OQ|^2) = (1 / |OM|^2) + (1 / |ON|^2) ∧
  (∀ m, m ∈ (-√(3), 0) ∪ (0, √(3)) -> n = √((15*m^2 + 180)/5))

-- Main statement: combines all individual questions
theorem math_proof_problem (k : ℝ) (G Q : point) (m n : ℝ): 
  question1 k → question2 G → question3 m n :=
by
  intros
  sorry

end math_proof_problem_l384_384019


namespace average_price_of_pair_l384_384609

def total_revenue : ℝ := 735
def number_of_pairs : ℝ := 75

theorem average_price_of_pair : total_revenue / number_of_pairs = 9.80 :=
by
  sorry

end average_price_of_pair_l384_384609


namespace terminating_decimal_integers_count_l384_384740

/-- For how many integer values of n between 1 and 180 inclusive does the decimal 
representation of n / 180 terminate? We need to calculate the number of integers n such 
that n includes at least 9 as a factor. The answer is 20. -/
theorem terminating_decimal_integers_count :
  { n : ℤ | 1 ≤ n ∧ n ≤ 180 ∧ 9 ∣ n }.to_finset.card = 20 := 
by sorry

end terminating_decimal_integers_count_l384_384740


namespace lloyd_normal_hours_l384_384487

-- Definitions based on the conditions
def regular_rate : ℝ := 3.50
def overtime_rate : ℝ := 1.5 * regular_rate
def total_hours_worked : ℝ := 10.5
def total_earnings : ℝ := 42
def normal_hours_worked (h : ℝ) : Prop := 
  h * regular_rate + (total_hours_worked - h) * overtime_rate = total_earnings

-- The theorem to prove
theorem lloyd_normal_hours : ∃ h : ℝ, normal_hours_worked h ∧ h = 7.5 := sorry

end lloyd_normal_hours_l384_384487


namespace exists_two_numbers_l384_384775

theorem exists_two_numbers (s : Finset ℝ) (h : s.card = 7) :
    ∃ x y ∈ s, x ≠ y ∧ 0 ≤ (x - y) / (1 + x * y) ∧ (x - y) / (1 + x * y) ≤ 1 / Real.sqrt 3 := 
by 
  sorry

end exists_two_numbers_l384_384775


namespace num_terminating_decimals_l384_384678

theorem num_terminating_decimals (n : ℕ) (h1 : 1 ≤ n ∧ n ≤ 180) :
  (∃ k : ℕ, k = 20) := by
  have h2 : ∀ n, (1 ≤ n ∧ n ≤ 180) → (∃ k, k = (180 / 9)) := sorry
  have h3 : 180 / 9 = 20 := by norm_num
  use 20
  exact h3

end num_terminating_decimals_l384_384678


namespace find_a_asymptotes_of_hyperbola_l384_384014

-- Define the parabola with focus
def parabola_focus (p: ℝ) : ℝ × ℝ := (-2, 0)

-- Define the hyperbola and its focus
def hyperbola_focus (a b: ℝ) : ℝ × ℝ := (a^2 + b^2, 0)

-- Given the parabola at focus (-2, 0) and the hyperbola defined as mentioned:
def hyperbola_asymptote (a: ℝ) : Prop := 
  ∃ (x y: ℝ), y = x ∨ y = -x

-- Prove that a = sqrt(2)
theorem find_a (a: ℝ) : 
  (parabola_focus (-2) = hyperbola_focus a a) → a = real.sqrt 2 :=
sorry

-- Prove the equations of the asymptotes are y = ±x
theorem asymptotes_of_hyperbola (a: ℝ) : 
  a = real.sqrt 2 → hyperbola_asymptote a :=
sorry

end find_a_asymptotes_of_hyperbola_l384_384014


namespace terminating_decimal_count_l384_384750

theorem terminating_decimal_count : 
  (Finset.filter (λ n, n % 3 = 0) (Finset.range 181)).card = 60 := by
  sorry

end terminating_decimal_count_l384_384750


namespace boys_to_admit_or_expel_l384_384551

-- Definitions from the conditions
def total_students : ℕ := 500

def girls_percent (x : ℕ) : ℕ := (x * total_students) / 100

-- Definition of the calculation under the new policy
def required_boys : ℕ := (total_students * 3) / 5

-- Main statement we need to prove
theorem boys_to_admit_or_expel (x : ℕ) (htotal : x + girls_percent x = total_students) :
  required_boys - x = 217 := by
  sorry

end boys_to_admit_or_expel_l384_384551


namespace length_AD_eq_19_2_l384_384844

noncomputable theory

variables (A B C D M X T Q : Type) 
          [rectangle A B C D] 
          [midpoint M B C]
          [on_line X B C]
          [equal_segments M X X]
          [right_angle AX XD]
          [line_perp T X B C]
          [equal_segments B X X T]
          [intersection Q XD XT]

variables (XA AQ XQ : ℝ)
          [XA_val : XA = 15]
          [AQ_val : AQ = 20]
          [XQ_val : XQ = 9]

theorem length_AD_eq_19_2 : length AD = 19.2 :=
by
  sorry

end length_AD_eq_19_2_l384_384844


namespace alpha_of_i_l384_384477

def imaginary_unit : ℂ := complex.i

def smallest_positive_integer (z : ℂ) : ℕ :=
  nat.find (exists_digit_travel : exists n : ℕ, 0 < n ∧ z^n = 1)

theorem alpha_of_i : smallest_positive_integer imaginary_unit = 4 :=
sorry

end alpha_of_i_l384_384477


namespace complex_conjugate_coordinates_l384_384070

noncomputable def complexConjugateCoordinates : ℂ :=
  complex.conj (10 * complex.I / (3 + complex.I))

theorem complex_conjugate_coordinates : complexConjugateCoordinates = 1 - 3 * complex.I := by
  sorry

end complex_conjugate_coordinates_l384_384070


namespace terminating_decimals_l384_384713

theorem terminating_decimals (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 180) :
  (∃ k : ℕ, k = 20 ∧ ∀ m : ℕ, 1 ≤ m ∧ m ≤ 180 → (decimal_terminates (m / 180) ↔ m % 9 = 0)) :=
by {
  sorry
}

def decimal_terminates (x : ℚ) : Prop :=
  sorry

end terminating_decimals_l384_384713


namespace centipede_socks_shoes_order_l384_384255

noncomputable def factorial (n : ℕ) : ℕ :=
if h : n = 0 then 1 else n * factorial (n - 1)

theorem centipede_socks_shoes_order :
  let total_items := 20 in
  let total_permutations := factorial total_items in
  let constraint_probability := (1 / 2) ^ 10 in
  (total_permutations / 2 ^ 10 = factorial 20 / 2 ^ 10) :=
begin
  let total_items := 20,
  let total_permutations := factorial total_items,
  let constraint_probability := (1 / 2) ^ 10,
  have h1 : total_permutations = factorial 20, from rfl,
  have h2 : constraint_probability = (1 / 2) ^ 10, from rfl,
  show total_permutations / 2 ^ 10 = factorial 20 / 2 ^ 10, from h1 ▸ h2 ▸ rfl
end

end centipede_socks_shoes_order_l384_384255


namespace LeonaEarnsGivenHourlyRate_l384_384533

theorem LeonaEarnsGivenHourlyRate :
  (∀ (c: ℝ) (t h e: ℝ), 
    (c = 24.75) → 
    (t = 3) → 
    (h = c / t) → 
    (e = h * 5) →
    e = 41.25) :=
by
  intros c t h e h1 h2 h3 h4
  sorry

end LeonaEarnsGivenHourlyRate_l384_384533


namespace nine_point_circle_angle_l384_384906

theorem nine_point_circle_angle {A B C : Point} 
  (h_acute : acute_triangle A B C)
  (H : Orthocenter A B C)
  (O : Circumcenter A B C)
  (N : Midpoint A B) 
  (CD : Altitude A B C)
  (nine_point_circle : ∃ N E D, 
                         OnNinePointCircle N E D 
                         ∧ Midpoint A B = N 
                         ∧ FootPerpendicular C A B = D 
                         ∧ Midpoint (Segment C H) = E) :
  ∃ center : Point, angle_at center N D = 2 * abs (angle A B - angle A C) :=
sorry

end nine_point_circle_angle_l384_384906


namespace divisors_of_90_l384_384036

def num_pos_divisors (n : ℕ) : ℕ :=
  let factors := if n = 90 then [(2, 1), (3, 2), (5, 1)] else []
  factors.foldl (fun acc (p, k) => acc * (k + 1)) 1

theorem divisors_of_90 : num_pos_divisors 90 = 12 := by
  sorry

end divisors_of_90_l384_384036


namespace number_of_positive_divisors_36_gt_1_l384_384820

theorem number_of_positive_divisors_36_gt_1 : 
  ∃ (n : ℕ), n = 36 ∧ (∀ (d : ℕ), d > 1 ∧ d ∣ n → list.length ([2, 3, 4, 6, 9, 12, 18, 36] : list ℕ) = 8) :=
  sorry

end number_of_positive_divisors_36_gt_1_l384_384820


namespace parallel_lines_and_planes_l384_384814

variables {a b : Type} {α β : Type} [linear_ordered_field a] [linear_ordered_field b] [affine_space α β]

-- Definitions: line and plane relationships
def is_subset (line: Type) (plane: Type) := sorry  -- Placeholder for relation "is a subset of"
def are_parallel (x y: Type) := sorry  -- Placeholder for relation "are parallel"

-- The mathematically equivalent proof problem
theorem parallel_lines_and_planes
  (a_in_beta : is_subset a β)
  (alpha_parallel_beta : are_parallel α β) :
  are_parallel a α :=
sorry

end parallel_lines_and_planes_l384_384814


namespace manufacturing_cost_of_shoe_l384_384169

theorem manufacturing_cost_of_shoe
  (transportation_cost_per_shoe : ℝ)
  (selling_price_per_shoe : ℝ)
  (gain_percentage : ℝ)
  (manufacturing_cost : ℝ)
  (H1 : transportation_cost_per_shoe = 5)
  (H2 : selling_price_per_shoe = 282)
  (H3 : gain_percentage = 0.20)
  (H4 : selling_price_per_shoe = (manufacturing_cost + transportation_cost_per_shoe) * (1 + gain_percentage)) :
  manufacturing_cost = 230 :=
sorry

end manufacturing_cost_of_shoe_l384_384169


namespace trig_identity_1_trig_identity_2_l384_384383

noncomputable def point := ℚ × ℚ

namespace TrigProblem

open Real

def point_on_terminal_side (α : ℝ) (p : point) : Prop :=
  let (x, y) := p
  ∃ r : ℝ, r = sqrt (x^2 + y^2) ∧ x/r = cos α ∧ y/r = sin α

theorem trig_identity_1 {α : ℝ} (h : point_on_terminal_side α (-4, 3)) :
  (sin (π / 2 + α) - cos (π + α)) / (sin (π / 2 - α) - sin (π - α)) = 8 / 7 :=
sorry

theorem trig_identity_2 {α : ℝ} (h : point_on_terminal_side α (-4, 3)) :
  sin α * cos α = -12 / 25 :=
sorry

end TrigProblem

end trig_identity_1_trig_identity_2_l384_384383


namespace different_numerators_count_l384_384875

-- Define the set of all rational numbers r of the form 0.ab repeated
def T : Set ℚ := { r | ∃ (a b : ℤ), (0 ≤ a) ∧ (a ≤ 9) ∧ (0 ≤ b) ∧ (b ≤ 9) ∧
                                      r = (10 * a + b) / 99 ∧ 0 < r ∧ r < 1 }

-- Define the predicate that checks if two integers are coprime
def coprime (m n : ℤ) : Prop := Int.gcd m n = 1

-- Define the top-level Lean theorem
theorem different_numerators_count :
  ∃ n : ℕ, n = 60 ∧ ∀ r ∈ T, ∃ p q : ℤ, coprime p q ∧ r = p / q ∧ q = 99 → p ∉ {1, 2, ..., 99} := sorry

end different_numerators_count_l384_384875


namespace largest_prime_factor_of_1023_l384_384567

/-- The largest prime factor of 1023 is 31. -/
theorem largest_prime_factor_of_1023 : ∃ p : ℕ, nat.prime p ∧ ∃ q : ℕ, 1023 = p * q ∧ (∀ r : ℕ, nat.prime r ∧ r ∣ 1023 → r ≤ p) := 
sorry

end largest_prime_factor_of_1023_l384_384567


namespace count_valid_numbers_l384_384821

-- Define a function that calculates the sum of the digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Define the necessary condition for the numbers
def is_valid_number (n : ℕ) : Prop :=
  n < 1000 ∧ n = 4 * sum_of_digits n

-- Define the proof statement
theorem count_valid_numbers : 
  (Finset.filter is_valid_number (Finset.range 1000)).card = 2 :=
by
  sorry

end count_valid_numbers_l384_384821


namespace statement_1_statement_2_statement_3_statement_4_number_of_correct_statements_l384_384312

variables {a b : ℝ^3}

def are_parallel (a b : ℝ^3) : Prop := ∃ k : ℝ, a = k • b

theorem statement_1 (a b : ℝ^3) : are_parallel a b → a = b := sorry

theorem statement_2 (a b : ℝ^3) : ∥a∥ = ∥b∥ → a = b := sorry

theorem statement_3 (a b : ℝ^3) : ∥a∥ = ∥b∥ → are_parallel a b := sorry

theorem statement_4 (a b : ℝ^3) : a = b → ∥a∥ = ∥b∥ :=
begin
  intro h,
  rw h,
end

-- Determine the number of correct statements
theorem number_of_correct_statements (a b : ℝ^3) : 
  (statement_1 a b → false ∧ statement_2 a b → false ∧ statement_3 a b → false ∧ statement_4 a b) →
  1 = 1 := by
begin
  intro h,
  exact rfl,
end

end statement_1_statement_2_statement_3_statement_4_number_of_correct_statements_l384_384312


namespace aaron_position_p100_l384_384287

def movement : ℕ → ℝ × ℝ
| 0 => (0, 0)
| 1 => (1, 0)
| n =>
  let position := movement (n - 1)
  let direction := (n - 1) % 4
  match direction with
  | 0 => (position.1 + 2, position.2)
  | 1 => (position.1, position.2 + 2)
  | 2 => (position.1 - 2, position.2)
  | _ => (position.1, position.2 - 2)

theorem aaron_position_p100 : movement 100 = (22, -6) := by
  sorry

end aaron_position_p100_l384_384287


namespace no_integers_satisfy_eq_l384_384907

theorem no_integers_satisfy_eq (m n : ℤ) : m^2 ≠ n^5 - 4 := 
by {
  sorry
}

end no_integers_satisfy_eq_l384_384907


namespace terminating_decimals_count_l384_384725

theorem terminating_decimals_count : 
  let N := 180 in
  let count_terminating_decimals := (List.range' 1 N).filter (λ n, n % 3 = 0) in
  count_terminating_decimals.length = 60 :=
by
  let N := 180
  let count_terminating_decimals := (List.range' 1 N).filter (λ n, n % 3 = 0)
  sorry

end terminating_decimals_count_l384_384725


namespace course_selection_l384_384535

theorem course_selection :
  let num_students := 5 in
  let at_least_2_students_choose_each_course := true in
  ∃ ways_to_choose_courses,
  ways_to_choose_courses = 20 := 
by 
  sorry

end course_selection_l384_384535


namespace find_range_of_a_l384_384888

noncomputable def range_of_a : Set ℝ :=
  {a | (∀ x : ℝ, x^2 - 2 * x > a) ∨ (∃ x0 : ℝ, x0^2 + 2 * a * x0 + 2 - a = 0)}

theorem find_range_of_a :
  {a : ℝ | (∀ x : ℝ, x^2 - 2 * x > a) ∨ (∃ x0 : ℝ, x0^2 + 2 * a * x0 + 2 - a = 0)} = 
  {a | (-2 < a ∧ a < -1) ∨ (1 ≤ a)} :=
by
  sorry

end find_range_of_a_l384_384888


namespace domain_of_f_l384_384325

noncomputable def f (x : ℝ) : ℝ := sqrt (-6 * x^2 - 7 * x + 15)

theorem domain_of_f :
  {x : ℝ | -6 * x^2 - 7 * x + 15 ≥ 0} = {x : ℝ | x ∈ Set.Iic 1 ∪ Set.Ici (5 / 2)} :=
by sorry

end domain_of_f_l384_384325


namespace albert_mary_age_ratio_l384_384616

def albert_age_ratios (A M B : ℕ) (h1 : B = 4) (h2 : A = 4 * B) (h3 : M = A - 8) : Prop :=
  A / M = 2

theorem albert_mary_age_ratio : ∃ A M B : ℕ, albert_age_ratios A M B 4 sorry sorry sorry :=
begin
  sorry,
end

end albert_mary_age_ratio_l384_384616


namespace inverse_of_matrix_l384_384328

noncomputable def my_matrix := matrix ([[5, -3], [-2, 1]])

theorem inverse_of_matrix :
  ∃ (M_inv : matrix ℕ ℕ ℝ), (my_matrix.det ≠ 0) ∧ (my_matrix * M_inv = 1) → M_inv = matrix ([[ -1, -3 ], [-2, -5 ]]) :=
by
  sorry

end inverse_of_matrix_l384_384328


namespace shaded_area_l384_384850

theorem shaded_area (r : ℝ) (h_circle_radius : r = 6) : 
  let quarter_circle_area := (1 / 4) * Real.pi * r^2,
      triangle_area := (1 / 2) * r * r,
      one_section_area := quarter_circle_area - triangle_area,
      total_shaded_area := 8 * one_section_area
  in total_shaded_area = 72 * Real.pi - 144 :=
by
  -- This is where the proof would go, but we'll use sorry to skip it for now.
  sorry

end shaded_area_l384_384850


namespace sum_of_k_values_l384_384986

theorem sum_of_k_values : 
  (∀ k, (∃ x, x^2 - 3 * x + 2 = 0 ∧ x^2 - 5 * x + k = 0) → k = 4 ∨ k = 6) → 
  (∑ k in {4, 6}, k) = 10 :=
by 
  sorry

end sum_of_k_values_l384_384986


namespace complex_number_purely_imaginary_l384_384422

theorem complex_number_purely_imaginary (m : ℝ) :
  (∃ (m : ℝ), (m^2 - m) + m * I = m * I) → m = 1 :=
begin
  sorry
end

end complex_number_purely_imaginary_l384_384422


namespace smallest_integer_for_perfect_square_l384_384593

-- Given condition: y = 2^3 * 3^2 * 4^6 * 5^5 * 7^8 * 8^3 * 9^10 * 11^11
def y : ℕ := 2^3 * 3^2 * 4^6 * 5^5 * 7^8 * 8^3 * 9^10 * 11^11

-- The statement to prove
theorem smallest_integer_for_perfect_square (y : ℕ) : ∃ n : ℕ, n = 110 ∧ ∃ m : ℕ, (y * n) = m^2 := 
by {
  sorry
}

end smallest_integer_for_perfect_square_l384_384593


namespace proof_problem_l384_384788

-- Define the propositions p and q.
def p (a : ℝ) : Prop := a < -1/2 

def q (a b : ℝ) : Prop := a > b → (1 / (a + 1)) < (1 / (b + 1))

-- Define the final proof problem: proving that "p or q" is true.
theorem proof_problem (a b : ℝ) : (p a) ∨ (q a b) := by
  sorry

end proof_problem_l384_384788


namespace shaded_area_l384_384340

theorem shaded_area (R : ℝ) (hR : 0 < R) :
  let area_equilateral_triangle := (R^2 * Real.sqrt 3) / 4 in
  let area_sector := (Real.pi * R^2) / 6 in
  let area_segment := area_sector - area_equilateral_triangle in
  let area_petal := 2 * area_segment in
  let total_area := 6 * area_petal in
  total_area = 2 * Real.pi * R^2 - 3 * R^2 * Real.sqrt 3 :=
begin
  let area_equilateral_triangle := (R^2 * Real.sqrt 3) / 4,
  let area_sector := (Real.pi * R^2) / 6,
  let area_segment := area_sector - area_equilateral_triangle,
  let area_petal := 2 * area_segment,
  let total_area := 6 * area_petal,
  sorry
end

end shaded_area_l384_384340


namespace arithmetic_geometric_ratio_l384_384469

theorem arithmetic_geometric_ratio
  (a : ℕ → ℤ) 
  (d : ℤ)
  (h_seq : ∀ n, a (n+1) = a n + d)
  (h_geometric : (a 3)^2 = a 1 * a 9)
  (h_nonzero_d : d ≠ 0) :
  a 11 / a 5 = 5 / 2 :=
by sorry

end arithmetic_geometric_ratio_l384_384469


namespace larger_part_of_separation_l384_384912

theorem larger_part_of_separation (x y : ℝ) (h1 : x + y = 66) (h2 : 0.40 * x = 0.625 * y + 10) : x = max x y := 
by
  have h3 : y = 66 - x := by linarith
  have h4 : 0.40 * x = 0.625 * (66 - x) + 10 := by rw [h3]; exact h2
  have h5 : 0.40 * x = 41.25 - 0.625 * x + 10 := by linarith
  have h6 : 1.025 * x = 51.25 := by linarith
  have h7 : x = 50 := by linarith
  have h8 : y = 66 - 50 := by linarith
  have h9 : y = 16 := by linarith
  have h10 : max x y = 50 := by simp [h7, h9]
  exact h10

end larger_part_of_separation_l384_384912


namespace vertices_count_leq_eight_l384_384965

-- Define a point in space with integer coordinates
structure Point3D :=
  (x : ℤ)
  (y : ℤ)
  (z : ℤ)

-- Define a convex polyhedron with vertices being a finite set of integer points
structure ConvexPolyhedron :=
  (vertices : set Point3D)
  (convex : ∀ A B ∈ vertices, ∀ t ∈ Icc (0 : ℚ) 1, Point3D.mk 
    (⌊t * A.x + (1 - t) * B.x⌋)
    (⌊t * A.y + (1 - t) * B.y⌋)
    (⌊t * A.z + (1 - t) * B.z⌋) ∈ vertices)
  (no_interior_integer_points : ∀ P, P ∉ vertices → ∀ A B ∈ vertices, ∀ t ∈ Ioc 0 1,
    Point3D.mk 
    (⌊t * A.x + (1 - t) * B.x⌋)
    (⌊t * A.y + (1 - t) * B.y⌋)
    (⌊t * A.z + (1 - t) * B.z⌋) ≠ P)
  (no_edge_integer_points : ∀ A B ∈ vertices, 
    ∀ t ∈ Ioo 0 1, Point3D.mk 
    (⌊t * A.x + (1 - t) * B.x⌋)
    (⌊t * A.y + (1 - t) * B.y⌋)
    (⌊t * A.z + (1 - t) * B.z⌋) ∉ vertices)

theorem vertices_count_leq_eight (P : ConvexPolyhedron) : 
  P.vertices.finite ∧ P.vertices.to_finset.card ≤ 8 :=
sorry

end vertices_count_leq_eight_l384_384965


namespace scientific_notation_l384_384944

theorem scientific_notation (x : ℝ) (hx : x = 0.000000007) : x = 7 * 10 ^ (-9) :=
by
  rw [hx]
  norm_num
  sorry

end scientific_notation_l384_384944


namespace common_ratio_unique_l384_384425

-- Define the conditions for the problem
variables {α : Type*} [Field α] {a : ℕ → α}
def geom_seq (a: ℕ → α) (q: α) : Prop := ∀ n, a(n+1) = q * a(n)

-- main theorem to prove
theorem common_ratio_unique (a : ℕ → ℝ) (q : ℝ) (h1 : a 0 + a 3 = 10) (h2 : a 1 + a 4 = 20) (h3 : geom_seq a q) :
  q = 2 :=
sorry -- proof omitted

end common_ratio_unique_l384_384425


namespace number_is_eight_l384_384413

theorem number_is_eight (x : ℤ) (h : x - 2 = 6) : x = 8 := 
sorry

end number_is_eight_l384_384413


namespace total_cards_correct_l384_384108

-- Define the number of dozens each person has
def dozens_per_person : Nat := 9

-- Define the number of cards per dozen
def cards_per_dozen : Nat := 12

-- Define the number of people
def num_people : Nat := 4

-- Define the total number of Pokemon cards in all
def total_cards : Nat := dozens_per_person * cards_per_dozen * num_people

-- The statement to be proved
theorem total_cards_correct : total_cards = 432 := 
by 
  -- Proof omitted as requested
  sorry

end total_cards_correct_l384_384108


namespace terminating_decimal_count_number_of_terminating_decimals_l384_384674

theorem terminating_decimal_count {n : ℕ} (h1 : 1 ≤ n) (h2 : n ≤ 180) :
  (∃ k, n = 9 * k ∧ 1 ≤ 9 * k ∧ 9 * k ≤ 180) ↔ n ∈ (finset.range 21).image (λ k, 9 * k) :=
by sorry

theorem number_of_terminating_decimals :
  (finset.range 21).card = 20 :=
by sorry

end terminating_decimal_count_number_of_terminating_decimals_l384_384674


namespace work_required_to_compress_spring_l384_384311

noncomputable def work_compression (k Δh : ℝ) : ℝ :=
  - (k * Δh^2) / 2

theorem work_required_to_compress_spring (k Δh : ℝ) :
  (∫ x in 0..Δh, -k * x) = - (k * Δh^2) / 2 :=
sorry

end work_required_to_compress_spring_l384_384311


namespace num_lines_through_P_and_hyperbola_single_intersection_l384_384173

theorem num_lines_through_P_and_hyperbola_single_intersection :
  let P := (4, 4)
  let hyperbola_eq := ∀ (x y : ℝ), x^2 / 16 - y^2 / 9 = 1
  ∃ (l : ℝ → ℝ), 
    (l 4 = 4) ∧ (∀ x, (l x)^2 / 16 - ((l x)^2 / 9) ≠ 1) :=
begin
  sorry
end

end num_lines_through_P_and_hyperbola_single_intersection_l384_384173


namespace sequence_product_value_l384_384646

theorem sequence_product_value :
  (1 / 3) * (9 / 1) * (1 / 27) * (81 / 1) * (1 / 243) * (729 / 1) * (1 / 729) * (2187 / 1) = 729 :=
by
  snorient_rm sorry

end sequence_product_value_l384_384646


namespace trips_required_l384_384632

def radius_cylinder := 12 -- radius of the cylindrical barrel in inches
def height_cylinder := 40 -- height of the cylindrical barrel in inches
def radius_sphere := 8 -- radius of the spherical bucket in inches

noncomputable def volume_cylinder := π * radius_cylinder^2 * height_cylinder
noncomputable def volume_sphere := (4/3) * π * radius_sphere^3

noncomputable def number_of_trips := (volume_cylinder / volume_sphere).ceil -- number of full trips required to fill the barrel

theorem trips_required : number_of_trips = 3 :=
by
  -- Proof omitted
  sorry

end trips_required_l384_384632


namespace proof_problem_l384_384382

variables {α : ℝ} {m : ℝ} {y := sqrt 2} 

-- conditions
def point_on_circle : Prop := (m, sqrt 2) ∈ { p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = 3 }

def in_first_quadrant : Prop := (m > 0) ∧ (sqrt 2 > 0)

-- questions and answers
def tan_alpha := y / m
def target_expr := (2 * cos (α / 2) ^ 2 - sin α - 1) / (sqrt 2 * sin (π / 4 + α))

-- theorem to be proved
theorem proof_problem (h1 : point_on_circle) (h2 : in_first_quadrant) : 
  (m = 1) →
  (tan_alpha = sqrt 2) →
  (target_expr = 2 * sqrt 2 - 3) :=
by
  intros
  sorry

end proof_problem_l384_384382


namespace circumscribed_circle_area_l384_384256

noncomputable def circumradius (s : ℝ) : ℝ := s / Real.sqrt 3
noncomputable def circle_area (r : ℝ) : ℝ := Real.pi * r^2

theorem circumscribed_circle_area (s : ℝ) (hs : s = 15) : circle_area (circumradius s) = 75 * Real.pi :=
by
  sorry

end circumscribed_circle_area_l384_384256


namespace battery_life_ge_6_l384_384963

-- Defining the service life of the battery as a normal random variable with mean 4
-- and variance sigma^2, where sigma > 0.

noncomputable theory
open ProbabilityTheory MeasureTheory

variable (σ : ℝ) (hσ : σ > 0)

-- X is a normally distributed random variable N(4, σ^2)
def X : ProbabilitySpace ℝ := Normal 4 σ

-- The probability that service life is at least 2 years is given as 0.9
axiom hx2 : ℙ {x | X x ≥ 2} = 0.9

-- Prove that the probability the battery lasts at least 6 years is 0.1
theorem battery_life_ge_6 : ℙ {x | X x ≥ 6} = 0.1 :=
sorry

end battery_life_ge_6_l384_384963


namespace china_math_olympiad_34_2023_l384_384481

-- Defining the problem conditions and verifying the minimum and maximum values of S.
theorem china_math_olympiad_34_2023 {a b c d e : ℝ}
  (h1 : a ≥ -1)
  (h2 : b ≥ -1)
  (h3 : c ≥ -1)
  (h4 : d ≥ -1)
  (h5 : e ≥ -1)
  (h6 : a + b + c + d + e = 5) :
  (-512 ≤ (a + b) * (b + c) * (c + d) * (d + e) * (e + a)) ∧
  ((a + b) * (b + c) * (c + d) * (d + e) * (e + a) ≤ 288) :=
sorry

end china_math_olympiad_34_2023_l384_384481
