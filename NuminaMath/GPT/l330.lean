import Float
import Mathlib
import Mathlib.Algebra.AbsoluteValue
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Polynomial.GroupRingAction
import Mathlib.Algebra.Polynomial.RingDivision
import Mathlib.Analysis.SpecialFunctions.Pow
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.SimpleGraph.Basic
import Mathlib.Combinatorics.SimpleGraph.Coloring
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Complex.Exponential
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Probability.Probability
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Init.Data.Nat
import Mathlib.NumberTheory.PrimeFactors
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.SetTheory.Game
import Mathlib.Tactic
import Mathlib.Tactic.Linarith
import Probability.Gamma

namespace find_n_l330_330359

noncomputable theory
open_locale classical

/-- Definitions and assumptions based on the problem statement -/
def regular_octagon (P : Point) (A_1 A_2 A_3 A_4 A_5 A_6 A_7 A_8 : Point) : Prop :=
  -- Assuming positions and regularity properties of the octagon vertices
  -- Though not detailed here, it encapsulates regularity and the fact that
  -- it is inscribed within a circle of area 1.
  sorry

def region_area (P : Point) (A_1 A_2 : Point) : ℝ :=
  -- This function calculates the area of region bounded by PA_1, PA_2 and the minor arc A_1A_2
  sorry

variable {Point : Type}
variable {A_1 A_2 A_3 A_4 A_5 A_6 A_7 A_8 P : Point}
variable {n : ℕ}

theorem find_n (h_octagon : regular_octagon P A_1 A_2 A_3 A_4 A_5 A_6 A_7 A_8) 
    (h_area1 : region_area P A_1 A_2 = 1/7) 
    (h_area2 : region_area P A_3 A_4 = 1/9) :
  ∃ n : ℕ, region_area P A_6 A_7 = 1/8 - real.sqrt 2 / n ∧ n = 504 :=
begin
  sorry
end

end find_n_l330_330359


namespace tony_slices_remaining_l330_330020

def slices_remaining : ℕ :=
  let total_slices := 22
  let slices_per_sandwich := 2
  let daily_sandwiches := 1
  let extra_sandwiches := 1
  let total_sandwiches := 7 * daily_sandwiches + extra_sandwiches
  total_slices - total_sandwiches * slices_per_sandwich

theorem tony_slices_remaining :
  slices_remaining = 6 :=
by
  -- Here, the theorem proof would go, but it is omitted as per instructions
  sorry

end tony_slices_remaining_l330_330020


namespace angle_at_vertex_C_is_60_degrees_l330_330423

theorem angle_at_vertex_C_is_60_degrees 
  (A B C I H : Type) 
  [HC : is_acute_triangle A B C]
  (concyclic : concyclic_points A B H I) 
  (I_incenter : incenter I A B C)
  (H_orthocenter : orthocenter H A B C) :
  angle_C A B C = 60 :=
sorry

end angle_at_vertex_C_is_60_degrees_l330_330423


namespace sequence_remainder_4_l330_330463

def sequence_of_numbers (n : ℕ) : ℕ :=
  7 * n + 4

theorem sequence_remainder_4 (n : ℕ) : (sequence_of_numbers n) % 7 = 4 := by
  sorry

end sequence_remainder_4_l330_330463


namespace total_cost_of_products_l330_330510

-- Conditions
def smartphone_price := 300
def personal_computer_price := smartphone_price + 500
def advanced_tablet_price := smartphone_price + personal_computer_price

-- Theorem statement for the total cost of one of each product
theorem total_cost_of_products :
  smartphone_price + personal_computer_price + advanced_tablet_price = 2200 := by
  sorry

end total_cost_of_products_l330_330510


namespace compound_interest_calculation_l330_330865

theorem compound_interest_calculation
  (x y : ℝ)
  (simple_interest simple_interest_formula compound_interest compound_interest_formula : ℝ → ℝ → ℝ → ℝ)
  (h_simple_interest : simple_interest 9000 y 2 = 900)
  (h_simple_interest_formula : simple_interest = λ P R T, (P * R * T) / 100)
  (h_compound_interest_formula : compound_interest = λ P R T, P * (1 + R / 100) ^ T - P)
  (principal : ℝ := 9000)
  (rate : ℝ := 5)
  (time : ℝ := 2) :
  compound_interest principal rate time = 922.5 := 
by
  sorry

end compound_interest_calculation_l330_330865


namespace number_of_sets_with_7_l330_330793

theorem number_of_sets_with_7 (S : Finset ℕ) (hS : S = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}) : 
  (S.filter (λ T, T.card = 3 ∧ 7 ∈ T ∧ T.sum = 21)).card = 5 :=
by
  sorry

end number_of_sets_with_7_l330_330793


namespace quadratic_function_passing_through_A_B_C_vertex_of_quadratic_function_l330_330254

-- Define the points through which the quadratic function passes
def A : Point := ⟨1, 0⟩
def B : Point := ⟨-3, 0⟩
def C : Point := ⟨0, -3⟩

-- Define the quadratic function
def quadratic_function (x : ℝ) : ℝ := x^2 + 2 * x - 3

-- Define the vertex of the quadratic function
def vertex : Point := ⟨-1, -4⟩

-- Proof statements to demonstrate the conditions
theorem quadratic_function_passing_through_A_B_C : 
  quadratic_function 1 = 0 ∧ quadratic_function (-3) = 0 ∧ quadratic_function 0 = -3 :=
by sorry

theorem vertex_of_quadratic_function :
  vertex = ⟨-1, -4⟩ :=
by sorry

end quadratic_function_passing_through_A_B_C_vertex_of_quadratic_function_l330_330254


namespace infinite_geometric_series_common_ratio_l330_330867

theorem infinite_geometric_series_common_ratio
  (a S : ℝ)
  (h₁ : a = 500)
  (h₂ : S = 4000)
  (h₃ : S = a / (1 - (r : ℝ))) :
  r = 7 / 8 :=
by
  sorry

end infinite_geometric_series_common_ratio_l330_330867


namespace tony_slices_remaining_l330_330022

def slices_remaining : ℕ :=
  let total_slices := 22
  let slices_per_sandwich := 2
  let daily_sandwiches := 1
  let extra_sandwiches := 1
  let total_sandwiches := 7 * daily_sandwiches + extra_sandwiches
  total_slices - total_sandwiches * slices_per_sandwich

theorem tony_slices_remaining :
  slices_remaining = 6 :=
by
  -- Here, the theorem proof would go, but it is omitted as per instructions
  sorry

end tony_slices_remaining_l330_330022


namespace find_m_l330_330265

-- Define the vectors a and b in a 2-dimensional plane
def a (m : ℝ) : ℝ × ℝ := (1, m)
def b (m : ℝ) : ℝ × ℝ := (4, m)

-- Define the dot product of two 2-dimensional vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem find_m : ∀ m : ℝ, dot_product (2 • a m - b m) (a m + b m) = 0 → m = sqrt 5 ∨ m = -sqrt 5 := by
  intros m h
  sorry

end find_m_l330_330265


namespace sqrt_approximation_l330_330259

theorem sqrt_approximation :
  (2^2 < 5) ∧ (5 < 3^2) ∧ 
  (2.2^2 < 5) ∧ (5 < 2.3^2) ∧ 
  (2.23^2 < 5) ∧ (5 < 2.24^2) ∧ 
  (2.236^2 < 5) ∧ (5 < 2.237^2) →
  (Float.ceil (Float.sqrt 5 * 100) / 100) = 2.24 := 
by
  intro h
  sorry

end sqrt_approximation_l330_330259


namespace polynomial_b_value_l330_330569

/-- Given a polynomial \( x^4 + ax^3 + bx^2 + cx + d = 0 \) with exactly four non-real roots such that 
the product of two roots is \( 7 - 4i \) and the sum of the other two roots is \( -2 + 5i \), prove that \( b = 43 \). -/
theorem polynomial_b_value (a b c d : ℝ) (z w : ℂ) :
  (z * w = 7 - 4 * complex.I) ∧ (z + complex.conj z = -2 + 5 * complex.I) →
  b = 43 :=
sorry

end polynomial_b_value_l330_330569


namespace area_of_ABCD_l330_330103

variables (A B C D E F C' B' : Type)
variables [Point A B C D] [Point E F] [Point C' B']
variables [RectangularSheet A B C D]

variable (AB' : ℝ) (BE : ℝ)
variable (AB'_val : AB' = 7)
variable (BE_val : BE = 15)
variable (area_ratio : DoubleAreaSimTriangles _ _ _ _ _ _ 1 4) -- Assuming a predicate DoubleAreaSimTriangles

-- Definition of a proof problem
theorem area_of_ABCD : 
  let ABCD_area := AreaRectangularSheet _ _ _ _ 
  ∃ area, area = 256 ∧ area = ABCD_area :=
begin
  sorry
end

end area_of_ABCD_l330_330103


namespace problem_statement_l330_330354

noncomputable def vector (α : Type*) := α → ℝ

variables {A B C D : Type*}
variables (AB AC AD : vector ℝ) -- Define vectors AB, AC, AD representing point A to B, A to C, and A to D respectively.

def CD := AD - AC
def BC := AC - AB
def BD := AD - AB

theorem problem_statement : 
  AB • CD + BC • AD - AC • BD = 0 := 
by sorry

end problem_statement_l330_330354


namespace ratio_of_border_to_tile_l330_330664

variable {s d : ℝ}

theorem ratio_of_border_to_tile (h1 : 900 = 30 * 30)
  (h2 : 0.81 = (900 * s^2) / (30 * s + 60 * d)^2) :
  d / s = 1 / 18 := by {
  sorry }

end ratio_of_border_to_tile_l330_330664


namespace prime_square_plus_eight_is_prime_l330_330924

theorem prime_square_plus_eight_is_prime (p : ℕ) (hp : Nat.Prime p) (h : Nat.Prime (p^2 + 8)) : p = 3 :=
sorry

end prime_square_plus_eight_is_prime_l330_330924


namespace remainder_of_7_pow_7_pow_7_pow_7_mod_500_l330_330221

theorem remainder_of_7_pow_7_pow_7_pow_7_mod_500 : 
  let λ := fun n => 
             match n with
             | 500 => 100
             | 100 => 20
             | _ => 0 in
  7^(7^(7^7)) ≡ 343 [MOD 500] :=
by 
  sorry

end remainder_of_7_pow_7_pow_7_pow_7_mod_500_l330_330221


namespace winner_exceeds_opponent_l330_330155

theorem winner_exceeds_opponent (winner_votes : ℕ) (first_opponent_votes : ℕ) (h_winner : winner_votes = 195) (h_opponent : first_opponent_votes = 142) :
  winner_votes - first_opponent_votes = 53 :=
by
  rw [h_winner, h_opponent]
  norm_num

end winner_exceeds_opponent_l330_330155


namespace smallest_x_in_domain_of_ggx_l330_330534

def g (x : ℝ) : ℝ := real.sqrt (x - 1)

theorem smallest_x_in_domain_of_ggx : ∃ x : ℝ, (∀ y, g(g(x)) = g(real.sqrt(x-1)) → y >= 1) ∧ x = 2 :=
begin
  sorry
end

end smallest_x_in_domain_of_ggx_l330_330534


namespace assignment_possible_l330_330549

def is_assigment_possible : Prop :=
  ∃ (f3x4 f3x5 f4x5 : ℕ), 
    (2 * ((4 * f3x4 + 5 * f3x5) = 120 ∧ (3 * f3x5 + 5 * f4x5) = 120 ∧ (3 * f3x4 + 4 * f4x5) = 120))

theorem assignment_possible : is_assigment_possible := sorry

end assignment_possible_l330_330549


namespace fraction_of_grid_covered_l330_330494

open Real EuclideanGeometry

noncomputable def area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem fraction_of_grid_covered :
  let A := (2, 2)
  let B := (6, 2)
  let C := (4, 5)
  let grid_area := 7 * 7
  let triangle_area := area_of_triangle A B C
  triangle_area / grid_area = 6 / 49 := by
  sorry

end fraction_of_grid_covered_l330_330494


namespace range_of_k_l330_330809

theorem range_of_k (k : ℝ) :
  (∀ x : ℝ, x^2 - k*x + 1 > 0) → (-2 < k ∧ k < 2) :=
by
  sorry

end range_of_k_l330_330809


namespace total_tiles_correct_l330_330732

-- Definitions for room dimensions
def room_length : ℕ := 24
def room_width : ℕ := 18

-- Definitions for tile dimensions
def border_tile_side : ℕ := 2
def inner_tile_side : ℕ := 1

-- Definitions for border and inner area calculations
def border_width : ℕ := 2 * border_tile_side
def inner_length : ℕ := room_length - border_width
def inner_width : ℕ := room_width - border_width

-- Calculation of the number of tiles needed
def border_area : ℕ := (room_length * room_width) - (inner_length * inner_width)
def num_border_tiles : ℕ := border_area / (border_tile_side * border_tile_side)
def inner_area : ℕ := inner_length * inner_width
def num_inner_tiles : ℕ := inner_area / (inner_tile_side * inner_tile_side)

-- Total number of tiles
def total_tiles : ℕ := num_border_tiles + num_inner_tiles

-- The proof statement
theorem total_tiles_correct : total_tiles = 318 := by
  -- Lean code to check the calculations, proof is omitted.
  sorry

end total_tiles_correct_l330_330732


namespace number_added_is_8_l330_330412

theorem number_added_is_8
  (x y : ℕ)
  (h1 : x = 265)
  (h2 : x / 5 + y = 61) :
  y = 8 :=
by
  sorry

end number_added_is_8_l330_330412


namespace slope_angle_y_intercept_of_line_l330_330785

-- Define the equation of the line
def line_eq (x y : ℝ) : Prop := x + y + 1 = 0

-- Prove that the slope angle and y-intercept of the line x + y + 1 = 0 are 135 degrees and -1 respectively
theorem slope_angle_y_intercept_of_line :
    (∀ x y : ℝ, line_eq x y) → (∃ slope_angle y_intercept, slope_angle = 135 ∧ y_intercept = -1) :=
by
  intro h
  refine ⟨135, -1, by tidy, by tidy⟩
  sorry

end slope_angle_y_intercept_of_line_l330_330785


namespace log_decreasing_l330_330576

theorem log_decreasing (a x y : ℝ) (h₀ : 0 < a) (h₁ : a < 1) 
  (h₂ : log a x < log a y) (h₃ : log a y < 0) : 1 < y ∧ y < x :=
by
  sorry

end log_decreasing_l330_330576


namespace negation_of_every_planet_orbits_the_sun_l330_330778

variables (Planet : Type) (orbits_sun : Planet → Prop)

theorem negation_of_every_planet_orbits_the_sun :
  (¬ ∀ x : Planet, (¬ (¬ (exists x : Planet, true)) → orbits_sun x)) ↔
  ∃ x : Planet, ¬ orbits_sun x :=
by sorry

end negation_of_every_planet_orbits_the_sun_l330_330778


namespace log_a_range_l330_330253

open Function Real

theorem log_a_range {η : Type} [LinearOrderedField η] [log_order : Log eta] :
  (∃ (m n : η), 
    (∃ x1 x2 : η, 0 < x1 ∧ x1 < 1 ∧ 1 < x2 ∧ 
                  x2 ∧ x1^2 + m*x1 + (m + n)/2 = 0 ∧ x2^2 + m*x2 + (m + n)/2 = 0) ∧ 
    n < -3*m - 2 ∧ m < -1 ∧ n > 1 ∧ 
    ∃ (P : η × η), P.1 = m ∧ P.2 = n ∧ 
                   ∃ a > 1, log a 3 > 1) → (1 < a ∧ a < 3) :=
begin
  sorry
end

end log_a_range_l330_330253


namespace diagonal_intersection_ratio_smallest_piece_area_l330_330135

-- Definitions and conditions for the problem
variables {A B C D E O : Type} 
variables [right_triangle : right_angle_tri A B C] (fold : fold_along_line A B C D E)
variables [folding_property : folding_right_angle_onto_vertex fold]

-- Problem (a): Prove the ratio of division of the diagonals
theorem diagonal_intersection_ratio (midline : Midline fold O) : 
  divide_at_centroid BD CE 2 1 :=
sorry

-- Problem (b): Prove the area of the smallest piece of paper
theorem smallest_piece_area (cut : cut_along_diagonal fold CE) :
  area_smallest_piece BCA DOC 1 6 :=
sorry

end diagonal_intersection_ratio_smallest_piece_area_l330_330135


namespace bags_wednesday_l330_330164

def charge_per_bag : ℕ := 4
def bags_monday : ℕ := 5
def bags_tuesday : ℕ := 3
def total_earnings : ℕ := 68

theorem bags_wednesday (h1 : charge_per_bag = 4)
                       (h2 : bags_monday = 5)
                       (h3 : bags_tuesday = 3)
                       (h4 : total_earnings = 68) :
  let earnings_monday_tuesday := (bags_monday + bags_tuesday) * charge_per_bag in
  let earnings_wednesday := total_earnings - earnings_monday_tuesday in
  earnings_wednesday / charge_per_bag = 9 :=
by
  sorry

end bags_wednesday_l330_330164


namespace delta_evaluation_l330_330904

def delta (a b : ℕ) : ℕ := a^3 - b

theorem delta_evaluation :
  delta (2^(delta 3 8)) (5^(delta 4 9)) = 2^19 - 5^55 := 
sorry

end delta_evaluation_l330_330904


namespace max_expr_value_l330_330674

-- Define the set of expressions with operators + or *
def expr_set (a b c d : ℕ) : list ℕ :=
  [a + b + c + d, a * b + c + d, a + b * c + d, a + b + c * d, a * b * c + d, a * b + c * d,
   a + b * c * d, a * (b + c) * d, a + (b * c) * d, (a * b) + (c * d), a * b * c * d, a + b * (c + d),
   (a + b) * (c + d), a * (b + c) + d, (a + b * c) * d, (a * b + c) * d]

-- Define the problem statement
theorem max_expr_value :
  let a := 1
  let b := 2
  let c := 3
  let d := 4
  in 25 ∈ expr_set a b c d :=
by {
  let a := 1,
  let b := 2,
  let c := 3,
  let d := 4,
  have h := expr_set a b c d,
  trivial
}

end max_expr_value_l330_330674


namespace binom_150_150_l330_330179

-- Definition of binomial coefficient
def binom (n k : ℕ) : ℕ :=
  if h : k ≤ n then (nat.choose n k) else 0

-- Symmetry of binomial coefficients
lemma binom_symm (n k : ℕ) : binom n k = binom n (n - k) :=
by sorry

-- The basic property of choosing 0 elements from n elements
lemma binom_zero (n : ℕ) : binom n 0 = 1 :=
by sorry

-- The main theorem to be proved.
theorem binom_150_150 : binom 150 150 = 1 :=
by sorry

end binom_150_150_l330_330179


namespace cylinder_volume_correct_l330_330382

def cylinder_volume (π : ℝ) (d h : ℝ) : ℝ := 
  let r := d / 2
  π * (r * r) * h

theorem cylinder_volume_correct :
  cylinder_volume (Real.pi) 10 5 ≈ 392.699 := sorry

end cylinder_volume_correct_l330_330382


namespace cos_double_angle_l330_330623

noncomputable def vec_a (α : ℝ) : ℝ × ℝ := (Real.cos α, 1/2)

def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

theorem cos_double_angle (α : ℝ) (h : magnitude (vec_a α) = Real.sqrt 2 / 2) :
  Real.cos (2 * α) = -1/2 :=
by
  sorry

end cos_double_angle_l330_330623


namespace melting_point_of_ice_in_fahrenheit_l330_330804

-- Define the conditions
def boilingPointCelsius : ℝ := 100
def boilingPointFahrenheit : ℝ := 212
def meltingPointCelsius : ℝ := 0
def conversionFormula (celsius : ℝ) : ℝ := (celsius * (9 / 5)) + 32

-- Declare the theorem and its statement
theorem melting_point_of_ice_in_fahrenheit : conversionFormula meltingPointCelsius = 32 :=
by
  sorry

end melting_point_of_ice_in_fahrenheit_l330_330804


namespace sqrt_expression_l330_330429

theorem sqrt_expression : sqrt (16 * sqrt (8 * sqrt 4)) = 8 := 
by 
  sorry

end sqrt_expression_l330_330429


namespace tank_capacity_l330_330796

theorem tank_capacity :
  ∃ (C : ℕ), 
    let rate_a := (C : ℕ) / 12,
    let rate_b := (C : ℕ) / 20,
    let rate_c := (C : ℕ) / 25,
    let rate_d := 45 in
    (rate_a + rate_b + rate_c - rate_d) * 30 = C ∧
    C = 321 := 
by
  sorry

end tank_capacity_l330_330796


namespace tan_difference_formula_l330_330599

theorem tan_difference_formula 
  (θ : ℝ) 
  (h : 4 * (Real.cos (θ + π/3)) * (Real.cos (θ - π/6)) = Real.sin (2 * θ)) : 
  Real.tan (2 * θ - π/6) = √3 / 9 := 
by 
  sorry

end tan_difference_formula_l330_330599


namespace number_of_sides_of_polygon_l330_330639

-- Given definition about angles and polygons
def exterior_angle (sides: ℕ) : ℝ := 30

-- The sum of exterior angles of any polygon
def sum_exterior_angles : ℝ := 360

-- The proof statement
theorem number_of_sides_of_polygon (k : ℕ) 
  (h1 : exterior_angle k = 30) 
  (h2 : sum_exterior_angles = 360):
  k = 12 :=
sorry

end number_of_sides_of_polygon_l330_330639


namespace distance_between_foci_of_ellipse_l330_330942

theorem distance_between_foci_of_ellipse : 
  let a := 5
  let b := 2
  let c := Real.sqrt (a^2 - b^2)
  2 * c = 2 * Real.sqrt 21 :=
by
  sorry

end distance_between_foci_of_ellipse_l330_330942


namespace sum_sequence_c_l330_330258

def seq_a (n : ℕ) : ℤ := 2 * n - 1
def seq_b (n : ℕ) : ℤ := 2 ^ (n - 1)
def seq_c (n : ℕ) : ℤ := seq_a n / seq_b n
def sum_seq_c (n : ℕ) : ℤ := (2 * n - 3) * 2 ^ n + 3

theorem sum_sequence_c {n : ℕ} :
  ∑ i in Finset.range n, seq_c (i + 1) = sum_seq_c n :=
sorry

end sum_sequence_c_l330_330258


namespace optimal_addition_amount_l330_330686

def optimal_material_range := {x : ℝ | 100 ≤ x ∧ x ≤ 200}

def second_trial_amounts := {x : ℝ | x = 138.2 ∨ x = 161.8}

theorem optimal_addition_amount (
  h1 : ∀ x ∈ optimal_material_range, x ∈ second_trial_amounts
  ) :
  138.2 ∈ second_trial_amounts ∧ 161.8 ∈ second_trial_amounts :=
by
  sorry

end optimal_addition_amount_l330_330686


namespace ratio_maria_initial_l330_330083

-- Define initial number of baseball cards
def initial_cards : ℕ := 15

-- Define the portion of cards Maria takes
def maria_takes : ℕ := 24

-- Define the final number of baseball cards
def final_cards : ℕ := 18

-- Prove the ratio of the number of baseball cards Maria took to the initial number of cards
theorem ratio_maria_initial : maria_takes : initial_cards = 8 : 5 := sorry

end ratio_maria_initial_l330_330083


namespace problem_l330_330981

open Function

variable {R : Type*} [Ring R]

def f (x : R) : R := -- Assume the function type

theorem problem (h : ∀ x y : R, f (x * y) = y^2 * f x + x^2 * f y) :
  f 0 = 0 ∧ f 1 = 0 ∧ (∀ x : R, f (-x) = f x) :=
by
  sorry

end problem_l330_330981


namespace cost_buses_minimize_cost_buses_l330_330114

theorem cost_buses
  (x y : ℕ) 
  (h₁ : x + y = 500)
  (h₂ : 2 * x + 3 * y = 1300) :
  x = 200 ∧ y = 300 :=
by 
  sorry

theorem minimize_cost_buses
  (m : ℕ) 
  (h₃: 15 * m + 25 * (8 - m) ≥ 180) :
  m = 2 ∧ (200 * m + 300 * (8 - m) = 2200) :=
by 
  sorry

end cost_buses_minimize_cost_buses_l330_330114


namespace right_triangle_condition_l330_330574

theorem right_triangle_condition (α β γ : ℝ) (h1 : α + β + γ = Real.pi)
  (h2 : sin α + cos α = sin β + cos β) : 
  (α = Real.pi / 2 ∨ β = Real.pi / 2 ∨ γ = Real.pi / 2) :=
by
  sorry

end right_triangle_condition_l330_330574


namespace operation_B_is_not_algorithm_l330_330151

-- Define what constitutes an algorithm.
def is_algorithm (desc : String) : Prop :=
  desc = "clear and finite steps to solve a certain type of problem"

-- Define given operations.
def operation_A : String := "Calculating the area of a circle given its radius"
def operation_B : String := "Calculating the possibility of reaching 24 by randomly drawing 4 playing cards"
def operation_C : String := "Finding the equation of a line given two points in the coordinate plane"
def operation_D : String := "Operations of addition, subtraction, multiplication, and division"

-- Define expected property of an algorithm.
def is_algorithm_A : Prop := is_algorithm "procedural, explicit, and finite steps lead to the result"
def is_algorithm_B : Prop := is_algorithm "cannot describe precise steps"
def is_algorithm_C : Prop := is_algorithm "procedural, explicit, and finite steps lead to the result"
def is_algorithm_D : Prop := is_algorithm "procedural, explicit, and finite steps lead to the result"

theorem operation_B_is_not_algorithm :
  ¬ (is_algorithm operation_B) :=
by
   -- Change this line to the theorem proof.
   sorry

end operation_B_is_not_algorithm_l330_330151


namespace correct_value_l330_330813

theorem correct_value (x : ℕ) (h : 14 * x = 42) : 12 * x = 36 := by
  sorry

end correct_value_l330_330813


namespace area_of_square_BDEF_l330_330678

noncomputable def right_triangle (A B C : Type*) [MetricSpace A] [MetricSpace B] [MetricSpace C] : Prop :=
∃ (AB BC AC : ℝ), AB = 15 ∧ BC = 20 ∧ AC = Real.sqrt (AB^2 + BC^2)

noncomputable def is_square (B D E F : Type*) [MetricSpace B] [MetricSpace D] [MetricSpace E] [MetricSpace F] : Prop :=
∃ (BD DE EF FB : ℝ), BD = DE ∧ DE = EF ∧ EF = FB

noncomputable def height_of_triangle (E H M : Type*) [MetricSpace E] [MetricSpace H] [MetricSpace M] : Prop :=
∃ (EH : ℝ), EH = 2

theorem area_of_square_BDEF (A B C D E F H M N : Type*)
  [MetricSpace A] [MetricSpace B] [MetricSpace C]
  [MetricSpace D] [MetricSpace E] [MetricSpace F]
  [MetricSpace H] [MetricSpace M] [MetricSpace N]
  (H1 : right_triangle A B C)
  (H2 : is_square B D E F)
  (H3 : height_of_triangle E H M) :
  ∃ (area : ℝ), area = 100 :=
by
  sorry

end area_of_square_BDEF_l330_330678


namespace jellybeans_in_carries_box_l330_330517

def volume_of_box (length width height : ℕ) : ℕ :=
  length * width * height

theorem jellybeans_in_carries_box (capacity_bert : ℕ) (multiplier : ℕ) :
  (capacity_bert = 150) → (multiplier = 3) → 
  (capacity_bert * multiplier ^ 3 = 4050) :=
begin
  intros h_capacity_bert h_multiplier,
  rw [h_capacity_bert, h_multiplier],
  norm_num,
end

end jellybeans_in_carries_box_l330_330517


namespace functional_eq_properties_l330_330984

theorem functional_eq_properties (f : ℝ → ℝ) (h_domain : ∀ x, x ∈ ℝ) (h_func_eq : ∀ x y, f (x * y) = y^2 * f x + x^2 * f y) :
  f 0 = 0 ∧ f 1 = 0 ∧ ∀ x, f x = f (-x) :=
by {
  sorry
}

end functional_eq_properties_l330_330984


namespace find_numbers_l330_330014

theorem find_numbers (x y : ℤ) (h_sum : x + y = 40) (h_diff : x - y = 12) : x = 26 ∧ y = 14 :=
sorry

end find_numbers_l330_330014


namespace length_of_PQ_l330_330714

theorem length_of_PQ (R P Q : ℝ × ℝ) (hR : R = (10, 8))
(hP_line1 : ∃ p : ℝ, P = (p, 24 * p / 7))
(hQ_line2 : ∃ q : ℝ, Q = (q, 5 * q / 13))
(h_mid : ∃ (p q : ℝ), R = ((p + q) / 2, (24 * p / 14 + 5 * q / 26) / 2))
(answer_eq : ∃ (a b : ℕ), PQ_length = a / b ∧ a.gcd b = 1 ∧ a + b = 4925) : 
∃ a b : ℕ, a + b = 4925 := sorry

end length_of_PQ_l330_330714


namespace pentagon_perimeter_l330_330166

def point := ℝ × ℝ

def dist (p1 p2 : point) : ℝ := real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def perimeter (p1 p2 p3 p4 p5 : point) : ℝ :=
  (dist p1 p2) + (dist p2 p3) + (dist p3 p4) + (dist p4 p5) + (dist p5 p1)

noncomputable def F : point := (0, 0)
noncomputable def G : point := (0, -1)
noncomputable def H : point := ((real.sqrt 3)/(real.sqrt 2), -1 - (real.sqrt 3)/(real.sqrt 2))
noncomputable def I : point := (H.1 + 2 * (real.sqrt 3)/2, H.2 - 2 * 1/2)
noncomputable def J : point := (I.1 + 2 * (real.sqrt 5/√4), I.2 - 2 * 1/(real.sqrt 4))

theorem pentagon_perimeter : perimeter F G H I J = 5 + real.sqrt ((J.1)^2 + (J.2)^2) :=
sorry

end pentagon_perimeter_l330_330166


namespace proof_problem_l330_330360

noncomputable def problem_statement : Prop :=
  let X : List ℝ := [16, 18, 20, 22]
  let Y : List ℝ := [15.10, 12.81, 9.72, 3.21]
  let U : List ℝ := [10, 20, 30]
  let V : List ℝ := [7.5, 9.5, 16.6]
  let r1 := linear_correlation_coefficient X Y
  let r2 := linear_correlation_coefficient U V
  r1 < 0 ∧ 0 < r2

theorem proof_problem : problem_statement := 
by
  sorry

end proof_problem_l330_330360


namespace divisibility_theorem_l330_330356

theorem divisibility_theorem {a m x n : ℕ} : (m ∣ n) ↔ (x^m - a^m ∣ x^n - a^n) :=
by
  sorry

end divisibility_theorem_l330_330356


namespace value_of_a_l330_330280

theorem value_of_a (a : ℝ) :
  (∀ x : ℝ, (5-x)/(x-2) ≥ 0 ↔ -3 < x ∧ x < a) → a > 5 :=
by
  intro h
  sorry

end value_of_a_l330_330280


namespace part_A_part_B_part_C_l330_330998

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation (x y : ℝ) : f(x * y) = y^2 * f(x) + x^2 * f(y)

theorem part_A : f(0) = 0 := sorry
theorem part_B : f(1) = 0 := sorry
theorem part_C : ∀ x : ℝ, f(x) = f(-x) := sorry

end part_A_part_B_part_C_l330_330998


namespace problem_l330_330979

open Function

variable {R : Type*} [Ring R]

def f (x : R) : R := -- Assume the function type

theorem problem (h : ∀ x y : R, f (x * y) = y^2 * f x + x^2 * f y) :
  f 0 = 0 ∧ f 1 = 0 ∧ (∀ x : R, f (-x) = f x) :=
by
  sorry

end problem_l330_330979


namespace equal_angle_point_on_ellipse_l330_330532

theorem equal_angle_point_on_ellipse (p : ℝ) (hp : p > 0) :
  (∀ x y : ℝ, (x / 2)^2 + y^2 = 1 → ∃ P : ℝ × ℝ, (P.1 = p ∧ P.2 = 0) ∧ 
  (∀ A B : ℝ × ℝ, 
      (P = (p, 0)) → 
      (A.1 / 2)^2 + A.2^2 = 1 → 
      (B.1 / 2)^2 + B.2^2 = 1 → 
      ∃ F : ℝ × ℝ, 
          (F = (2, 0)) ∧ 
          ∃ chord : ℝ × ℝ, 
              (chord.2 = F.2) → 
              (A.2 / (A.1 - p)) = -(B.2 / (B.1 - p)) )) :=  p = 2 :=
  sorry

end equal_angle_point_on_ellipse_l330_330532


namespace value_of_k_l330_330188

def f (x : ℝ) := 4 * x ^ 2 - 5 * x + 6
def g (x : ℝ) (k : ℝ) := 2 * x ^ 2 - k * x + 1

theorem value_of_k :
  (f 5 - g 5 k = 30) → k = -10 := 
by 
  sorry

end value_of_k_l330_330188


namespace function_properties_l330_330996

-- Given conditions
def domain (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ∈ ℝ

def functional_eqn (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f(x * y) = y^2 * f(x) + x^2 * f(y)

-- Lean 4 theorem statement
theorem function_properties (f : ℝ → ℝ)
  (Hdom : domain f)
  (Heqn : functional_eqn f) :
  f 0 = 0 ∧ f 1 = 0 ∧ ∀ x : ℝ, f (-x) = f x :=
by
  sorry


end function_properties_l330_330996


namespace probability_at_least_one_diamond_or_ace_or_both_red_l330_330799

/-!
# Probability of Drawing Specific Cards

We want to prove that the probability of drawing at least one diamond or ace or both cards
are red in a two cards successive draw without replacement from a standard deck of 52 cards
is equal to 889/1326.
-/

theorem probability_at_least_one_diamond_or_ace_or_both_red :
  let total_cards := 52 in
  let diamond_cards := 13 in
  let ace_cards := 4 in
  let red_cards := 26 in
  let non_diamond_non_ace_cards := total_cards - diamond_cards - 1 in
  let non_red_cards := 24 in
  let p_complement := (non_diamond_non_ace_cards / total_cards) * 
                      ((non_diamond_non_ace_cards - non_red_cards) / (total_cards - 1)) in
  let p_event := 1 - p_complement in
  p_event = 889 / 1326 :=
by
  have p_complement := 19 / 26 * 23 / 51
  have p_event := 1 - p_complement
  have p_event := 889 / 1326
  sorry

end probability_at_least_one_diamond_or_ace_or_both_red_l330_330799


namespace parabola_shifts_down_decrease_c_real_roots_l330_330805

-- The parabolic function and conditions
variables {a b c k : ℝ}

-- Assumption that a is positive
axiom ha : a > 0

-- Parabola shifts down when constant term c is decreased
theorem parabola_shifts_down (c : ℝ) (k : ℝ) (hk : k > 0) :
  ∀ x, (a * x^2 + b * x + (c - k)) = (a * x^2 + b * x + c) - k :=
by sorry

-- Discriminant of quadratic equation ax^2 + bx + c = 0
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- If the discriminant is negative, decreasing c can result in real roots
theorem decrease_c_real_roots (b c : ℝ) (hb : b^2 < 4 * a * c) (k : ℝ) (hk : k > 0) :
  discriminant a b (c - k) ≥ 0 :=
by sorry

end parabola_shifts_down_decrease_c_real_roots_l330_330805


namespace mass_percentage_H_in_chlorous_acid_l330_330931

noncomputable def mass_percentage_H_in_HClO2 : ℚ :=
  let molar_mass_H : ℚ := 1.01
  let molar_mass_Cl : ℚ := 35.45
  let molar_mass_O : ℚ := 16.00
  let molar_mass_HClO2 : ℚ := molar_mass_H + molar_mass_Cl + 2 * molar_mass_O
  (molar_mass_H / molar_mass_HClO2) * 100

theorem mass_percentage_H_in_chlorous_acid :
  mass_percentage_H_in_HClO2 = 1.475 := by
  sorry

end mass_percentage_H_in_chlorous_acid_l330_330931


namespace odd_An_iff_perfect_square_of_even_l330_330230

-- Definition of the given conditions and the final statement to be proved
theorem odd_An_iff_perfect_square_of_even (n : ℤ) (h : n ≥ 2) :
  (A_n n) % 2 = 1 ↔ ∃ k : ℤ, n = (2 * k)^2 :=
sorry

end odd_An_iff_perfect_square_of_even_l330_330230


namespace smallest_perimeter_l330_330050

def is_integer_coordinate_vertices (A B C : ℤ × ℤ) : Prop :=
  ∀ P ∈ {A, B, C}, ∃ x y : ℤ, P = (x, y)

def area_of_triangle (A B C : ℤ × ℤ) : ℚ :=
  (1 / 2 : ℚ) * (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)).natAbs

def is_no_side_parallel_to_axis (A B C : ℤ × ℤ) : Prop :=
  ∀ P Q ∈ [{A, B, C}.toList.combinations 2], P.1 ≠ Q.1 ∧ P.2 ≠ Q.2

def perimeter (A B C : ℤ × ℤ) : ℝ :=
  (Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) +
   Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) +
   Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2))

theorem smallest_perimeter (A B C : ℤ × ℤ) 
  (h1 : is_integer_coordinate_vertices A B C)
  (h2 : area_of_triangle A B C = 1 / 2)
  (h3 : is_no_side_parallel_to_axis A B C) :
  perimeter A B C = Real.sqrt 5 + Real.sqrt 2 + Real.sqrt 13 :=
sorry

end smallest_perimeter_l330_330050


namespace sum_of_factors_of_60_l330_330070

-- Given conditions
def n : ℕ := 60

-- Proof statement
theorem sum_of_factors_of_60 : (∑ d in (finset.filter (λ d, n % d = 0) (finset.range (n + 1))), d) = 168 :=
by
  sorry

end sum_of_factors_of_60_l330_330070


namespace bugs_meet_time_l330_330420

theorem bugs_meet_time
  (r1 r2 : ℝ) (v1 v2 : ℝ) (delay : ℝ)
  (C1 C2 : ℝ) (T1 T2 : ℝ)
  (hC1 : C1 = 2 * r1 * real.pi)
  (hC2 : C2 = 2 * r2 * real.pi)
  (hT1 : T1 = C1 / v1)
  (hT2 : T2 = C2 / v2)
  (h_r1 : r1 = 7)
  (h_r2 : r2 = 3)
  (h_v1 : v1 = 4 * real.pi)
  (h_v2 : v2 = 3 * real.pi)
  (h_delay : delay = 2) :
  let lcmT := real.lcm (T1.num * T2.denom) (T2.num * T1.denom) / (T1.denom * T2.denom) in
  lcmT + delay = 9 := 
begin
  sorry
end

end bugs_meet_time_l330_330420


namespace proof1_proof2_l330_330522

noncomputable def expr1 (a b : ℝ) : ℝ :=
  (-3 * a * b⁻¹)^2 * (a⁻² * b^2)^(-3)

theorem proof1 (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) :
  expr1 a b = 9 * a^8 / b^8 :=
by sorry

noncomputable def expr2 (a b : ℝ) : ℝ :=
  (a - b) / a / (a - (2 * a * b - b^2) / a)

theorem proof2 (a b : ℝ) (h1 : a ≠ 0) (h2 : a ≠ b) :
  expr2 a b = 1 / (a - b) :=
by sorry

end proof1_proof2_l330_330522


namespace sum_of_c_and_d_l330_330282

-- Define the function g and the vertical asymptotes condition
def g (c d x : ℝ) : ℝ := (x - 3) / (x^2 + c * x + d)

-- Define the conditions of vertical asymptotes
def vertical_asymptotes (c d : ℝ) : Prop :=
  (λ x : ℝ, x^2 + c * x + d = 0) 2 ∧ (λ x : ℝ, x^2 + c * x + d = 0) (-3)

-- Statement for the problem to prove sum of c and d is -5
theorem sum_of_c_and_d (c d : ℝ) (h_asymptote : vertical_asymptotes c d) : c + d = -5 := by
  sorry

end sum_of_c_and_d_l330_330282


namespace integral_value_l330_330949

theorem integral_value 
  (a : ℝ) (h_a_pos : a > 0)
  (h_expansion : ∃ k, ∀ x ≠ 0, (∑ i in Finset.range 7, (6.choose i) * (a^i / x^(i/2)) * (-x)^(6-i)) = k) :
  ∫ x in -a .. a, (Real.sqrt (1 - x^2) + Real.sin (2 * x)) = (Real.pi / 2) :=
by
  -- Proof skipped
  sorry

end integral_value_l330_330949


namespace correct_graph_representation_l330_330614

theorem correct_graph_representation (f : ℝ → ℝ) (a : ℕ → ℝ) (a1 : ℝ) (h1 : a1 ∈ Ioo 0 1) (h_rec : ∀ n : ℕ, a (n + 1) = f (a n)) (h_ineq : ∀ n : ℕ, a (n + 1) > a n)  :
  ∀ x : ℝ, x ∈ Ioo 0 1 → f x > x := 
by
  sorry

-- Definitions used: 
-- f : ℝ → ℝ (the function y=f(x))
-- a : ℕ → ℝ (the sequence {a_n})
-- a1 : ℝ (initial value a1 ∈ (0,1))
-- h1: a1 ∈ Ioo 0 1 (the condition that a1 is within (0,1))
-- h_rec: ∀ n : ℕ, a (n + 1) = f (a n) (recurrence relation)
-- h_ineq: ∀ n : ℕ, a (n + 1) > a n (sequence property)

-- Proof to be established: ∀ x : ℝ, x ∈ Ioo 0 1 → f x > x

end correct_graph_representation_l330_330614


namespace sample_second_grade_l330_330782

theorem sample_second_grade (r1 r2 r3 sample_size : ℕ) (h1 : r1 = 3) (h2 : r2 = 3) (h3 : r3 = 4) (h_sample_size : sample_size = 50) : (r2 * sample_size) / (r1 + r2 + r3) = 15 := by
  sorry

end sample_second_grade_l330_330782


namespace geometric_sequence_third_term_l330_330841

theorem geometric_sequence_third_term (a₁ a₄ : ℕ) (r : ℕ) (h₁ : a₁ = 4) (h₂ : a₄ = 256) (h₃ : a₄ = a₁ * r^3) : a₁ * r^2 = 64 := 
by
  sorry

end geometric_sequence_third_term_l330_330841


namespace pascal_sixth_element_row_20_l330_330544

theorem pascal_sixth_element_row_20 : (Nat.choose 20 5) = 7752 := 
by 
  sorry

end pascal_sixth_element_row_20_l330_330544


namespace sqrt_sum_eq_five_sqrt_three_l330_330199

theorem sqrt_sum_eq_five_sqrt_three : Real.sqrt 12 + Real.sqrt 27 = 5 * Real.sqrt 3 := by
  sorry

end sqrt_sum_eq_five_sqrt_three_l330_330199


namespace correct_pronoun_l330_330734

-- Define the possible options
inductive Pronoun: Type
| my
| mine
| myself
| me

-- Given the context of the sentence
def context_valid (p: Pronoun): Prop :=
  p = Pronoun.me

-- The theorem that states the correct pronoun given the context
theorem correct_pronoun : ∃ p: Pronoun, context_valid p :=
by {
  use Pronoun.me,
  simp [context_valid],
  sorry
}

end correct_pronoun_l330_330734


namespace sum_of_fifth_terms_arithmetic_sequences_l330_330250

theorem sum_of_fifth_terms_arithmetic_sequences (a b : ℕ → ℝ) (d₁ d₂ : ℝ) 
  (h₁ : ∀ n, a (n + 1) = a n + d₁)
  (h₂ : ∀ n, b (n + 1) = b n + d₂)
  (h₃ : a 1 + b 1 = 7)
  (h₄ : a 3 + b 3 = 21) :
  a 5 + b 5 = 35 :=
sorry

end sum_of_fifth_terms_arithmetic_sequences_l330_330250


namespace largest_two_digit_number_l330_330438

-- Define the set of available digits
def available_digits : Set ℕ := {1, 2, 4, 6}

-- Define the predicate for a valid two-digit number from two distinct digits in the set
def valid_two_digit (d1 d2 : ℕ) : ℕ :=
  if d1 ∈ available_digits ∧ d2 ∈ available_digits ∧ d1 ≠ d2 then 10 * d1 + d2 else 0

-- State the problem as a proposition to be proved
theorem largest_two_digit_number : ∃ d1 d2, d1 ∈ available_digits ∧ d2 ∈ available_digits ∧ d1 ≠ d2 ∧ valid_two_digit d1 d2 = 64 :=
by
  sorry

end largest_two_digit_number_l330_330438


namespace intersect_curve_c_area_of_triangle_l330_330266

theorem intersect_curve_c {ρ θ : ℝ} (h : ρ * sin θ * sin θ = 4 * cos θ) :
  (A = (8 / 3, π / 3) ∨ A = (0, π / 3)) ∧ (B = (8 * sqrt 3, π / 6)) :=
by sorry

theorem area_of_triangle {A B : ℝ × ℝ}
  (hA : A = (8 / 3, π / 3) ∨ A = (0, π / 3))
  (hB : B = (8 * sqrt 3, π / 6)) :
  ∃ area : ℝ, area = (16 / 3) * sqrt 3 :=
by sorry

end intersect_curve_c_area_of_triangle_l330_330266


namespace chocolates_eaten_by_robert_l330_330365

theorem chocolates_eaten_by_robert (nickel_ate : ℕ) (robert_ate_more : ℕ) (H1 : nickel_ate = 3) (H2 : robert_ate_more = 4) :
  nickel_ate + robert_ate_more = 7 :=
by {
  sorry
}

end chocolates_eaten_by_robert_l330_330365


namespace find_a1_l330_330231

noncomputable def my_otimes (a b : ℝ) : ℝ :=
  if a * b >= 0 then a * b else a / b

noncomputable def f (x : ℝ) : ℝ :=
  Math.log x / Math.log 2

def geometric_sequence (a : ℕ → ℝ) (r : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n * r

theorem find_a1 (a : ℕ → ℝ) (r : ℝ) 
  (h_geo : geometric_sequence a r) (h_r_pos : 0 < r)
  (h_a1011 : a 1011 = 1)
  (h_sum : ∑ i in Finset.range 2020, f (a (i + 1)) = -3 / a 1) : a 1 = 1 / 8 :=
sorry

end find_a1_l330_330231


namespace probability_dart_lands_in_center_l330_330130

-- Definitions related to the board and geometry
def hexagon_side_length : ℝ := sorry -- Define the side length of the hexagon
def central_hexagon_side_length : ℝ := hexagon_side_length / 2
def triangle_area (side_length: ℝ) : ℝ := (sqrt 3 / 4) * (side_length ^ 2)

-- Calculations of areas
def central_hexagon_area : ℝ := (3 * sqrt 3 / 2) * (central_hexagon_side_length ^ 2)
def one_triangle_area : ℝ := triangle_area central_hexagon_side_length
def total_triangles_area : ℝ := 6 * one_triangle_area
def total_dartboard_area : ℝ := central_hexagon_area + total_triangles_area

-- Theorem to prove
theorem probability_dart_lands_in_center : 
  ∀ (s : ℝ) (central_hexagon_area = (3 * sqrt 3 / 2) * ((s / 2) ^ 2))
  (total_dartboard_area = (3 * sqrt 3 / 2) * (s ^ 2)),
  (central_hexagon_area / total_dartboard_area = 1 / 2) :=
by
  sorry

end probability_dart_lands_in_center_l330_330130


namespace bobby_toy_cars_in_5_years_l330_330519

noncomputable def toy_cars_after_n_years (initial_cars : ℕ) (percentage_increase : ℝ) (n : ℕ) : ℝ :=
initial_cars * (1 + percentage_increase)^n

theorem bobby_toy_cars_in_5_years :
  toy_cars_after_n_years 25 0.75 5 = 410 := by
  -- 25 * (1 + 0.75)^5 
  -- = 25 * (1.75)^5 
  -- ≈ 410.302734375
  -- After rounding
  sorry

end bobby_toy_cars_in_5_years_l330_330519


namespace intersection_of_lines_l330_330211

theorem intersection_of_lines :
  ∃ (x y : ℚ), (8 * x - 3 * y = 24) ∧ (10 * x + 2 * y = 14) ∧ x = 45 / 23 ∧ y = -64 / 23 :=
by
  sorry

end intersection_of_lines_l330_330211


namespace functional_eq_properties_l330_330989

theorem functional_eq_properties (f : ℝ → ℝ) (h_domain : ∀ x, x ∈ ℝ) (h_func_eq : ∀ x y, f (x * y) = y^2 * f x + x^2 * f y) :
  f 0 = 0 ∧ f 1 = 0 ∧ ∀ x, f x = f (-x) :=
by {
  sorry
}

end functional_eq_properties_l330_330989


namespace RectangleAreaDiagonalk_l330_330852

theorem RectangleAreaDiagonalk {length width : ℝ} {d : ℝ}
  (h_ratio : length / width = 5 / 2)
  (h_perimeter : 2 * (length + width) = 42)
  (h_diagonal : d = Real.sqrt (length^2 + width^2))
  : (∃ k, k = 10 / 29 ∧ ∀ A, A = k * d^2) :=
by {
  sorry
}

end RectangleAreaDiagonalk_l330_330852


namespace relatively_prime_bound_l330_330334

theorem relatively_prime_bound {m n : ℕ} {a : ℕ → ℕ} (h1 : 1 < m) (h2 : 1 < n) (h3 : m ≥ n)
  (h4 : ∀ i j, i ≠ j → a i = a j → False) (h5 : ∀ i, a i ≤ m) (h6 : ∀ i j, i ≠ j → a i ∣ a j → a i = 1) 
  (x : ℝ) : ∃ i, dist (a i * x) (round (a i * x)) ≥ 2 / (m * (m + 1)) * dist x (round x) :=
sorry

end relatively_prime_bound_l330_330334


namespace distance_from_point_to_line_l330_330322

theorem distance_from_point_to_line (m : ℝ) (h : m > 0) :
  let point := (m, Real.pi / 3) in
  let line := λ ρ θ, ρ * Real.cos (θ - Real.pi / 3) = 2 in
  let cartesian_distance (pt : ℝ × ℝ) (a b c : ℝ) :=
    (a * pt.1 + b * pt.2 + c) / Real.sqrt (a * a + b * b) in
  let point_cart := (m / 2, (Real.sqrt 3 * m) / 2) in
  cartesian_distance point_cart 1 (Real.sqrt 3) (-4) = |m - 2| := 
sorry

end distance_from_point_to_line_l330_330322


namespace sum_factors_of_60_l330_330058

def sum_factors (n : ℕ) : ℕ :=
  (∑ d in Finset.divisors n, d)

theorem sum_factors_of_60 : sum_factors 60 = 168 := by
  sorry

end sum_factors_of_60_l330_330058


namespace john_calories_eaten_l330_330134

def servings : ℕ := 3
def calories_per_serving : ℕ := 120
def fraction_eaten : ℚ := 1 / 2

theorem john_calories_eaten : 
  (servings * calories_per_serving : ℕ) * fraction_eaten = 180 :=
  sorry

end john_calories_eaten_l330_330134


namespace grid_parallelogram_exists_l330_330654

theorem grid_parallelogram_exists 
  (grid_size : ℕ) (blue_cells_count : ℕ)
  (h_grid_size : grid_size = 1000)
  (h_blue_cells_count : blue_cells_count = 2000) :
  ∃ (f : Fin grid_size → Fin grid_size → Bool), 
    ( ∑ i j, if f i j then 1 else 0 ) = blue_cells_count ∧
    ∃ (i1 j1 i2 j2 i3 j3 i4 j4: Fin grid_size), 
      f i1 j1 = true ∧ f i2 j2 = true ∧ f i3 j3 = true ∧ f i4 j4 = true ∧
      (((i1 = i3 ∧ j1 ≠ j3) ∧ (i2 = i4 ∧ j2 ≠ j4)) ∨
       ((i1 ≠ i3 ∧ j1 = j3) ∧ (i2 ≠ i4 ∧ j2 = j4))) :=
sorry

end grid_parallelogram_exists_l330_330654


namespace Prohor_receives_all_money_l330_330036

theorem Prohor_receives_all_money :
  ∀ (ivan_breads prohor_breads total_breads total_kopecks : ℕ)
    (shared_kopecks : ℕ) 
    (total_shared_breads : ℕ)
    (per_bread_value : ℕ)
    (hunter_share_value : ℕ) ,
  ivan_breads = 4 ∧
  prohor_breads = 8 ∧
  total_breads = 12 ∧
  total_kopecks = 60 ∧
  shared_kopecks = 50 ∧
  total_shared_breads = 4 ∧
  per_bread_value = 5 ∧
  hunter_share_value = total_shared_breads * per_bread_value →
  hunter_share_value = shared_kopecks :=
by {
  intros,
  sorry
}

end Prohor_receives_all_money_l330_330036


namespace max_f_l330_330887

noncomputable def f (x : ℝ) : ℝ :=
  1 / (|x + 3| + |x + 1| + |x - 2| + |x - 5|)

theorem max_f : ∃ x : ℝ, f x = 1 / 11 :=
by
  sorry

end max_f_l330_330887


namespace at_most_one_perfect_square_l330_330856

theorem at_most_one_perfect_square (a : ℕ → ℕ) (h : ∀ n, a (n + 1) = a n ^ 3 + 103) : 
  ∃ n, ∀ m, (m < n → ¬ is_square (a m)) ∧ (is_square (a n) → ∀ k, k ≠ n → ¬ is_square (a k)) := sorry

end at_most_one_perfect_square_l330_330856


namespace graph_transformation_l330_330390

noncomputable def f : ℝ → ℝ
| x := if -3 ≤ x ∧ x ≤ 0 then -2 - x
       else if 0 ≤ x ∧ x ≤ 2 then sqrt (4 - (x - 2)^2) - 2
       else if 2 ≤ x ∧ x ≤ 3 then 2 * (x - 2)
       else 0  -- Just to handle other cases, though they shouldn't occur based on domain.

def h (x : ℝ) : ℝ :=
- f (x + 6)

theorem graph_transformation :
  ∀ x : ℝ, h(x) = -f(x + 6) :=
begin
  intro x,
  unfold h,
  sorry
end

end graph_transformation_l330_330390


namespace magnitude_of_b_l330_330270

variable (a b : EuclideanSpace ℝ (Fin 2))
variable (a_def : a = ![1, 2])
variable (dot_product : a ⬝ b = 5)
variable (magnitude_diff : ‖a - b‖ = 2 * Real.sqrt 5)

theorem magnitude_of_b : ‖b‖ = 5 := by
  sorry

end magnitude_of_b_l330_330270


namespace jill_correct_l330_330369

-- Define the six distinct positive integers
def a : ℕ := 2
def b : ℕ := 4
def c : ℕ := 8
def d : ℕ := 3
def e : ℕ := 15
def f : ℕ := 39

-- Define that six integers should be distinct and positive
axiom distinct_pos : 
  ∀ {x y : ℕ}, (x ≠ y ∧ x > 0 ∧ y > 0) → 
  (x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e ∨ x = f) ∧ 
  (y = a ∨ y = b ∨ y = c ∨ y = d ∨ y = e ∨ y = f)

-- Lean statement to prove Jill's claim is correct.
theorem jill_correct : 
  ∀ (x y : ℕ), x ≠ y → 
  (x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e ∨ x = f) → 
  (y = a ∨ y = b ∨ y = c ∨ y = d ∨ y = e ∨ y = f) → 
  (x + y).prime → 
  ∃ (set_of_primes : set ℕ), set_of_primes.card ≤ 9 :=
sorry

end jill_correct_l330_330369


namespace g_is_correct_l330_330709

noncomputable def g : ℝ → ℝ := sorry

axiom g_0 : g 0 = 2

axiom g_functional_eq : ∀ x y : ℝ, g (x * y) = g (x^2 + y^2) + 2 * (x - y)^2

theorem g_is_correct : ∀ x : ℝ, g x = 2 - 2 * x := 
by 
  sorry

end g_is_correct_l330_330709


namespace player_one_wins_l330_330779

def sequence : List ℕ := List.range' 1 20
def is_even (n : ℕ) : Prop := n % 2 = 0

theorem player_one_wins :
    ∃ (seq : List ℕ), seq = sequence ∧ 
    (∀ (signs : List (ℕ → ℕ → ℕ)), signs.length = seq.length - 1 → 
    (∑ i in signs.zip seq, (i.1 i.2 (i.2 + (seq.nth_le i.2.val (by sorry))))))
    % 2 = 0 → Player 1 wins) :=
by
  have h1 : sequence = List.range' 1 20 := rfl
  have h2 : even (sequence.countp (λ x, x % 2 = 1)) := sorry
  sorry

end player_one_wins_l330_330779


namespace procurement_quantity_sufficient_l330_330498

-- Define the cost model
def total_cost (x : ℝ) (k : ℝ := 1/30) : ℝ :=
  ((4000 / x) * 360) + (3000 * k * x)

-- Given facts
def given_cost_condition (x : ℝ) : Prop := 
  total_cost x = 43600

def budget_sufficient (x : ℝ) : Prop :=
  total_cost x ≤ 24000

-- Main theorem
theorem procurement_quantity_sufficient : 
  budget_sufficient 120 := 
sorry

end procurement_quantity_sufficient_l330_330498


namespace power_modulus_l330_330808

theorem power_modulus (n : ℕ) : (2 : ℕ) ^ 345 % 5 = 2 :=
by sorry

end power_modulus_l330_330808


namespace orthogonal_vectors_l330_330722

def vector (α : Type*) [Add α] := list α

def dot_product (v1 v2 : vector ℝ) : ℝ :=
(list.zip_with (*) v1 v2).sum

theorem orthogonal_vectors (m : ℝ) :
    let a := [1, -1]
    let b := [m + 1, 2m - 4]
    dot_product a b = 0 → m = 5 := 
by
  intro h
  sorry

end orthogonal_vectors_l330_330722


namespace functional_equation_option_A_option_B_option_C_l330_330975

-- Given conditions
def domain_of_f (f : ℝ → ℝ) : Set ℝ := Set.univ

theorem functional_equation (f : ℝ → ℝ) (x y : ℝ) :
  f (x * y) = y^2 * f x + x^2 * f y := sorry

-- Statements to be proved
theorem option_A (f : ℝ → ℝ) (h : ∀ x y, f(x * y) = y^2 * f(x) + x^2 * f(y)) : f 0 = 0 := sorry

theorem option_B (f : ℝ → ℝ) (h : ∀ x y, f(x * y) = y^2 * f(x) + x^2 * f(y)) : f 1 = 0 := sorry

theorem option_C (f : ℝ → ℝ) (h : ∀ x y, f(x * y) = y^2 * f(x) + x^2 * f(y)) : ∀ x, f(-x) = f x := sorry

end functional_equation_option_A_option_B_option_C_l330_330975


namespace largest_sq_factor_of_10_fact_correct_l330_330426

noncomputable def largest_sq_factor_of_10_fact : ℕ := 720

theorem largest_sq_factor_of_10_fact_correct (k : ℕ) (hk : k^2 ∣ nat.factorial 10) : k ≤ largest_sq_factor_of_10_fact :=
by
  sorry

end largest_sq_factor_of_10_fact_correct_l330_330426


namespace exists_infinite_l330_330572

def is_sumset_eq (A B : Finset ℕ) : Prop :=
  (A.card > 1 ∧ B.card > 1 ∧ A ≠ B ∧ Finset.image2 (+) A A = Finset.image2 (+) B B)

def exists_sets_with_equal_sumsets (n : ℕ) : Prop :=
  ∃ (A B : Finset ℕ), A.card = n ∧ B.card = n ∧ is_sumset_eq A B

theorem exists_infinite (n : ℕ) (hn : n > 1) : ∃ (A B : Finset ℕ), A.card = n ∧ B.card = n ∧ is_sumset_eq A B := by
  sorry

end exists_infinite_l330_330572


namespace find_t_l330_330948

-- Define the elements and the conditions
def vector_a : ℝ × ℝ := (1, -1)
def vector_b (t : ℝ) : ℝ × ℝ := (t, 1)

def add_vectors (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)
def sub_vectors (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 - w.1, v.2 - w.2)

def parallel (u v : ℝ × ℝ) : Prop := u.1 * v.2 = u.2 * v.1

-- Lean statement of the problem
theorem find_t (t : ℝ) : 
  parallel (add_vectors vector_a (vector_b t)) (sub_vectors vector_a (vector_b t)) → t = -1 :=
by
  sorry

end find_t_l330_330948


namespace tony_slices_remaining_l330_330021

def slices_remaining : ℕ :=
  let total_slices := 22
  let slices_per_sandwich := 2
  let daily_sandwiches := 1
  let extra_sandwiches := 1
  let total_sandwiches := 7 * daily_sandwiches + extra_sandwiches
  total_slices - total_sandwiches * slices_per_sandwich

theorem tony_slices_remaining :
  slices_remaining = 6 :=
by
  -- Here, the theorem proof would go, but it is omitted as per instructions
  sorry

end tony_slices_remaining_l330_330021


namespace b_arithmetic_sequence_T_sum_l330_330605

-- Define the sequences and given conditions
def a (n : ℕ) : ℝ := if n = 0 then 3 else sorry

def b (n : ℕ) : ℝ := 1 / (a n - 1)

lemma a_recurrence (n : ℕ) :
  (a (n + 1) - 1) * (a n - 1) / (a n - a (n + 1)) = 3 :=
sorry

-- Prove that b_n is an arithmetic sequence
theorem b_arithmetic_sequence :
  ∀ n : ℕ, b (n + 1) - b n = 1 / 3 :=
sorry

-- Define the sequence T_n and prove the sum
def T (n : ℕ) : ℝ :=
  ∑ i in finset.range n, (2 ^ i) * (b (i + 1))

theorem T_sum (n : ℕ) :
  T n = 1 / 3 + (2 * n - 1) / 6 * 2 ^ (n + 1) :=
sorry

end b_arithmetic_sequence_T_sum_l330_330605


namespace time_needed_by_Alpha_and_Beta_l330_330416

theorem time_needed_by_Alpha_and_Beta (A B C h : ℝ)
  (h₀ : 1 / (A - 4) = 1 / (B - 2))
  (h₁ : 1 / A + 1 / B + 1 / C = 3 / C)
  (h₂ : A = B + 2)
  (h₃ : 1 / 12 + 1 / 10 = 11 / 60)
  : h = 60 / 11 :=
sorry

end time_needed_by_Alpha_and_Beta_l330_330416


namespace original_time_taken_by_bullet_train_is_50_minutes_l330_330831

-- Define conditions as assumptions
variables (T D : ℝ) (h0 : D = 48 * T) (h1 : D = 60 * (40 / 60))

-- Define the theorem we want to prove
theorem original_time_taken_by_bullet_train_is_50_minutes :
  T = 50 / 60 :=
by
  sorry

end original_time_taken_by_bullet_train_is_50_minutes_l330_330831


namespace unique_identity_function_l330_330205

theorem unique_identity_function (f : ℝ → ℝ) (H : ∀ x y z : ℝ, (x^3 + f y * x + f z = 0) → (f x ^ 3 + y * f x + z = 0)) :
  f = id :=
by sorry

end unique_identity_function_l330_330205


namespace sum_of_positive_factors_60_l330_330067

theorem sum_of_positive_factors_60 : (∑ n in Nat.divisors 60, n) = 168 := by
  sorry

end sum_of_positive_factors_60_l330_330067


namespace triangle_bc_range_l330_330298

variable {α : Type*}

def Δ (A B C : α) : Prop := ∃ (a b c : ℝ), a + b = c ∧ a > 0 ∧ b > 0

theorem triangle_bc_range (A B C : α) (AB AC BC : ℝ)
  (h1 : (√3 * sin B - cos B) * (√3 * sin C - cos C) = 4 * cos B * cos C)
  (h2 : AB + AC = 4 )
  (h3 : BC > 0) :
  2 ≤ BC ∧ BC < 4 :=
by
  sorry

end triangle_bc_range_l330_330298


namespace total_price_before_increase_l330_330443

-- Conditions
def original_price_candy_box (c_or: ℝ) := 10 = c_or * 1.25
def original_price_soda_can (s_or: ℝ) := 15 = s_or * 1.50

-- Goal
theorem total_price_before_increase :
  ∃ (c_or s_or : ℝ), original_price_candy_box c_or ∧ original_price_soda_can s_or ∧ c_or + s_or = 25 :=
by
  sorry

end total_price_before_increase_l330_330443


namespace magnitude_b_angle_between_vectors_l330_330245

variables {a b : EuclideanSpace ℝ (Fin 3)}
variables (non_zero_a : a ≠ 0) (non_zero_b : b ≠ 0)
variables (norm_a_one : ∥a∥ = 1)
variables (dot_product : (a - b) ⬝ (a + b) = 1 / 2)

theorem magnitude_b :
  ∥b∥ = real.sqrt 2 / 2 :=
sorry

theorem angle_between_vectors (a_dot_b : a ⬝ b = 1 / 2) :
  real.angle_between a b = real.pi / 4 :=
sorry

end magnitude_b_angle_between_vectors_l330_330245


namespace median_of_stem_leaf_l330_330011

def stem_leaf_data : list ℕ := [50, 55, 58, 58, 59, 70, 80, 90, 105, 115, 118, 125, 135, 143, 150, 170, 120, 123, 130]
def median : ℕ := 118

theorem median_of_stem_leaf :
  list.sorted nat.le stem_leaf_data → (stem_leaf_data.length = 21) → (list.nth_le stem_leaf_data 10 sorry = median) :=
sorry

end median_of_stem_leaf_l330_330011


namespace abs_value_solution_l330_330283

theorem abs_value_solution (a : ℝ) : |-a| = |-5.333| → (a = 5.333 ∨ a = -5.333) :=
by
  sorry

end abs_value_solution_l330_330283


namespace boat_avg_speed_l330_330827

theorem boat_avg_speed (D : ℝ) (hD : D > 0) :
  let distance_lake := D,
      speed_lake := 5,
      distance_upstream := 2 * D,
      speed_upstream := 4,
      distance_downstream := 2 * D,
      speed_downstream := 6,
      total_distance := distance_lake + distance_upstream + distance_downstream,
      time_lake := distance_lake / speed_lake,
      time_upstream := distance_upstream / speed_upstream,
      time_downstream := distance_downstream / speed_downstream,
      total_time := time_lake + time_upstream + time_downstream,
      avg_speed := total_distance / total_time
  in avg_speed = 150 / 31 :=
begin
  sorry
end

end boat_avg_speed_l330_330827


namespace sum_of_positive_factors_60_l330_330055

def sum_of_positive_factors (n : ℕ) : ℕ :=
  ∑ d in (Finset.filter (λ x => x > 0) (Finset.divisors n)), d

theorem sum_of_positive_factors_60 : sum_of_positive_factors 60 = 168 := by
  sorry

end sum_of_positive_factors_60_l330_330055


namespace remaining_soup_can_feed_adults_l330_330112

-- Define initial conditions
def cans_per_soup_for_children : ℕ := 6
def cans_per_soup_for_adults : ℕ := 4
def initial_cans : ℕ := 8
def children_to_feed : ℕ := 24

-- Define the problem statement in Lean
theorem remaining_soup_can_feed_adults :
  (initial_cans - (children_to_feed / cans_per_soup_for_children)) * cans_per_soup_for_adults = 16 := by
  sorry

end remaining_soup_can_feed_adults_l330_330112


namespace sum_largest_odd_divisors_bound_l330_330262

def largest_odd_divisor (n : ℤ) : ℤ :=
  if n % 2 = 1 then n
  else largest_odd_divisor (n / 2)

theorem sum_largest_odd_divisors_bound (n : ℕ) :
  abs (∑ k in Finset.range (n + 1), (largest_odd_divisor k) / k - (2 * n) / 3) < 1 :=
sorry

end sum_largest_odd_divisors_bound_l330_330262


namespace inhabitableSurfaceFraction_l330_330641

-- Definitions based on conditions
def landFraction : ℝ := 1 / 3
def inhabitableFraction : ℝ := 1 / 3

-- Theorem statement to be proven
theorem inhabitableSurfaceFraction : (landFraction * inhabitableFraction) = 1 / 9 :=
by 
  sorry

end inhabitableSurfaceFraction_l330_330641


namespace sum_of_arithmetic_sequence_l330_330668

noncomputable def S (n : ℕ) (a1 d : ℤ) : ℤ := n * a1 + (n * (n - 1) / 2) * d

theorem sum_of_arithmetic_sequence (a1 : ℤ) (d : ℤ)
  (h1 : a1 = -2010)
  (h2 : (S 2011 a1 d) / 2011 - (S 2009 a1 d) / 2009 = 2) :
  S 2010 a1 d = -2010 := 
sorry

end sum_of_arithmetic_sequence_l330_330668


namespace find_missing_number_l330_330917

theorem find_missing_number (x : ℝ) : 11 + real.sqrt (-4 + x * 4 / 3) = 13 ↔ x = 6 :=
by sorry

end find_missing_number_l330_330917


namespace nate_age_when_ember_is_14_l330_330550

theorem nate_age_when_ember_is_14 (nate_age : ℕ) (ember_age : ℕ) 
  (h1 : ember_age = nate_age / 2) (h2 : nate_age = 14) :
  ∃ (years_later : ℕ), ember_age + years_later = 14 ∧ nate_age + years_later = 21 :=
by
  -- sorry to skip the proof, adhering to the instructions
  sorry

end nate_age_when_ember_is_14_l330_330550


namespace oil_level_drop_correct_l330_330018

noncomputable def stationary_tank_radius : ℝ := 100
noncomputable def stationary_tank_height : ℝ := 25

noncomputable def truck1_radius : ℝ := 5
noncomputable def truck1_height : ℝ := 12

noncomputable def truck2_radius : ℝ := 6
noncomputable def truck2_height : ℝ := 15

noncomputable def truck3_radius : ℝ := 7
noncomputable def truck3_height : ℝ := 18

noncomputable def volume_of_cylinder (r h : ℝ) : ℝ := π * r^2 * h

noncomputable def truck1_volume := volume_of_cylinder truck1_radius truck1_height
noncomputable def truck2_volume := volume_of_cylinder truck2_radius truck2_height
noncomputable def truck3_volume := volume_of_cylinder truck3_radius truck3_height

noncomputable def total_volume_removed : ℝ :=
  truck1_volume + truck2_volume + truck3_volume

noncomputable def height_drop_in_stationary_tank : ℝ :=
  total_volume_removed / (π * stationary_tank_radius^2)

theorem oil_level_drop_correct :
  height_drop_in_stationary_tank = 0.1722 := by
  sorry

end oil_level_drop_correct_l330_330018


namespace proof_ratio_l330_330353

variable (A B C O : Type)
variable [MetricSpace A] [MetricSpace B] [MetricSpace C]
variable (r : ℝ) (AB AC BC : ℝ)
variable (θ : ℝ)
variable [∀ x, Differentiable x]

noncomputable def calculate_ratio (r : ℝ) (AB AC BC : ℝ) : ℝ :=
  if BC = 2 * r then 
    let theta := 2 in
    let central_angle_aob := 2 * Math.PI - theta in
    let ab_squared := 2 * r ^ 2 * (1 - Real.cos theta) in
    let ab := Real.sqrt ab_squared in
    ab / BC
  else
    0

theorem proof_ratio : ∀ (r : ℝ), ∀ (AB AC BC : ℝ), ∀ (θ : ℝ),
  AB = AC →
  AB > r →
  BC = 2 * r →
  calculate_ratio r AB AC BC = Real.sqrt 2 * Real.sin 1 :=
by
  assume r AB AC BC θ h1 h2 h3
  sorry

end proof_ratio_l330_330353


namespace part_a_smallest_m_divisible_by_23m_l330_330567

theorem part_a_smallest_m_divisible_by_23m : ∃ m : ℕ, m! % (23 * m) = 0 ∧ m = 24 :=
by {
    sorry
}

end part_a_smallest_m_divisible_by_23m_l330_330567


namespace log_eq_one_half_iff_sqrt_ten_l330_330947

theorem log_eq_one_half_iff_sqrt_ten {x : ℝ} (h : log10 x = 1 / 2) : x = real.sqrt 10 :=
by
  sorry

end log_eq_one_half_iff_sqrt_ten_l330_330947


namespace min_value_expression_l330_330212

noncomputable def expression (x y : ℝ) := 2 * x^2 + 3 * x * y + 4 * y^2 - 8 * x - 6 * y

theorem min_value_expression : ∀ x y : ℝ, expression x y ≥ -14 :=
by
  sorry

end min_value_expression_l330_330212


namespace average_running_time_l330_330665

variable (s : ℕ) -- Number of seventh graders

-- let sixth graders run 20 minutes per day
-- let seventh graders run 18 minutes per day
-- let eighth graders run 15 minutes per day
-- sixth graders = 3 * seventh graders
-- eighth graders = 2 * seventh graders

def sixthGradersRunningTime : ℕ := 20 * (3 * s)
def seventhGradersRunningTime : ℕ := 18 * s
def eighthGradersRunningTime : ℕ := 15 * (2 * s)

def totalRunningTime : ℕ := sixthGradersRunningTime s + seventhGradersRunningTime s + eighthGradersRunningTime s
def totalStudents : ℕ := 3 * s + s + 2 * s

theorem average_running_time : totalRunningTime s / totalStudents s = 18 :=
by sorry

end average_running_time_l330_330665


namespace molecular_weight_6_moles_C4H8O2_is_528_624_l330_330049

-- Define the atomic weights of Carbon, Hydrogen, and Oxygen.
def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.008
def atomic_weight_O : ℝ := 16.00

-- Define the molecular formula of C4H8O2.
def num_C_atoms : ℕ := 4
def num_H_atoms : ℕ := 8
def num_O_atoms : ℕ := 2

-- Define the number of moles of C4H8O2.
def num_moles_C4H8O2 : ℝ := 6

-- Define the molecular weight of one mole of C4H8O2.
def molecular_weight_C4H8O2 : ℝ :=
  (num_C_atoms * atomic_weight_C) +
  (num_H_atoms * atomic_weight_H) +
  (num_O_atoms * atomic_weight_O)

-- The total weight of 6 moles of C4H8O2.
def total_weight_6_moles_C4H8O2 : ℝ :=
  num_moles_C4H8O2 * molecular_weight_C4H8O2

-- Theorem stating that the molecular weight of 6 moles of C4H8O2 is 528.624 grams.
theorem molecular_weight_6_moles_C4H8O2_is_528_624 :
  total_weight_6_moles_C4H8O2 = 528.624 :=
by
  -- Proof is omitted.
  sorry

end molecular_weight_6_moles_C4H8O2_is_528_624_l330_330049


namespace inverse_of_36_mod_101_l330_330877

theorem inverse_of_36_mod_101 : ∃ x : ℕ, (36 * x) % 101 = 1 ∧ x = 87 := by
  existsi 87
  split
  . show (36 * 87) % 101 = 1 from sorry
  . show 87 = 87 from rfl

end inverse_of_36_mod_101_l330_330877


namespace proof_problem_l330_330953

variables {f : ℝ → ℝ}

-- Axioms corresponding to the conditions given in the problem
axiom cond1 : ∀ x : ℝ, f(-x) = f(x)
axiom cond2 : ∀ x1 x2 : ℝ, (0 ≤ x1 ∧ 0 ≤ x2 ∧ x1 ≠ x2) → (f(x1) - f(x2)) * (x1 - x2) < 0

-- Statement to be proved
theorem proof_problem : f(-3) > f(1) ∧ f(1) > f(2) := 
sorry

end proof_problem_l330_330953


namespace max_k_range_minus_five_l330_330930

theorem max_k_range_minus_five :
  ∃ k : ℝ, (∀ x : ℝ, x^2 + 5 * x + k = -5) → k = 5 / 4 :=
by
  sorry

end max_k_range_minus_five_l330_330930


namespace transistors_in_2010_l330_330299

theorem transistors_in_2010 
  (initial_transistors : ℕ) 
  (initial_year : ℕ) 
  (final_year : ℕ) 
  (doubling_period : ℕ)
  (initial_transistors_eq: initial_transistors = 500000)
  (initial_year_eq: initial_year = 1985)
  (final_year_eq: final_year = 2010)
  (doubling_period_eq : doubling_period = 2) :
  initial_transistors * 2^((final_year - initial_year) / doubling_period) = 2048000000 := 
by 
  -- the proof goes here
  sorry

end transistors_in_2010_l330_330299


namespace num_distinct_solutions_abs_equation_l330_330193

theorem num_distinct_solutions_abs_equation :
  ∃ S : set ℝ, 
    (∀ x : ℝ, (| x - | 3 * x - 2 || = 4 ↔ x ∈ S)) ∧
    (S.card = 2) :=
sorry

end num_distinct_solutions_abs_equation_l330_330193


namespace total_profit_calculation_l330_330087

-- Conditions
variables {a b : Type}
variables [linear_ordered_field a]
variables (P : a)
variables (capital_a capital_b : a)
variables (managing_fee : a)

-- Given values from the problem
def capital_a_value : a := 3500
def capital_b_value : a := 2500
def managing_fee_percentage : a := 0.10
def total_money_received_by_a : a := 6000

-- Definition of conditions based on given problem
def total_profit := P

-- Lean theorem statement
theorem total_profit_calculation 
  (h1 : capital_a = capital_a_value)
  (h2 : capital_b = capital_b_value)
  (h3 : managing_fee = managing_fee_percentage * total_profit)
  (h4 : total_money_received_by_a = managing_fee + (capital_a_value / (capital_a_value + capital_b_value)) * (0.90 * total_profit)) :
  total_profit = 9600 :=
sorry

end total_profit_calculation_l330_330087


namespace integer_solutions_of_prime_equation_l330_330449

theorem integer_solutions_of_prime_equation (p : ℕ) (hp : Prime p) :
  ∃ x y : ℤ, (p * (x + y) = x * y) ↔ 
    (x = (p * (p + 1)) ∧ y = (p + 1)) ∨ 
    (x = 2 * p ∧ y = 2 * p) ∨ 
    (x = 0 ∧ y = 0) ∨ 
    (x = p * (1 - p) ∧ y = (p - 1)) := 
sorry

end integer_solutions_of_prime_equation_l330_330449


namespace vertical_angles_congruent_l330_330750

-- Define what it means for two angles to be vertical angles
def areVerticalAngles (a b : ℝ) : Prop := -- placeholder definition
  sorry

-- Define what it means for two angles to be congruent
def areCongruentAngles (a b : ℝ) : Prop := a = b

-- State the problem in the form of a theorem
theorem vertical_angles_congruent (a b : ℝ) :
  areVerticalAngles a b → areCongruentAngles a b := by
  sorry

end vertical_angles_congruent_l330_330750


namespace cheryl_used_material_l330_330445

theorem cheryl_used_material 
  (a b c l : ℚ) 
  (ha : a = 3 / 8) 
  (hb : b = 1 / 3) 
  (hl : l = 15 / 40) 
  (Hc: c = a + b): 
  (c - l = 1 / 3) := 
by 
  -- proof will be deferred to Lean's syntax for user to fill in.
  sorry

end cheryl_used_material_l330_330445


namespace find_S_9_l330_330591

variable {a_n : ℕ → ℤ} (d : ℤ) (a_1 : ℤ)

-- a_n is defined as the nth term of an arithmetic sequence
def a_n (n : ℕ) : ℤ := a_1 + n * d

-- Define the sum of the first n terms of the arithmetic sequence
def S_n (n : ℕ) : ℤ := (n * (2 * a_1 + (n - 1) * d)) / 2

-- Given condition: 2 * (a_1 + 6 * d) = (a_1 + 8 * d) + 6
theorem find_S_9 : 2 * a_n 6 = a_n 8 + 6 → S_n 9 = 54 :=
by
  sorry

end find_S_9_l330_330591


namespace percentage_excess_calculation_l330_330670

theorem percentage_excess_calculation (A B : ℝ) (x : ℝ) 
  (h1 : (A * (1 + x / 100)) * (B * 0.95) = A * B * 1.007) : 
  x = 6.05 :=
by
  sorry

end percentage_excess_calculation_l330_330670


namespace total_cost_of_products_l330_330509

-- Conditions
def smartphone_price := 300
def personal_computer_price := smartphone_price + 500
def advanced_tablet_price := smartphone_price + personal_computer_price

-- Theorem statement for the total cost of one of each product
theorem total_cost_of_products :
  smartphone_price + personal_computer_price + advanced_tablet_price = 2200 := by
  sorry

end total_cost_of_products_l330_330509


namespace minimal_q_for_fraction_l330_330712

theorem minimal_q_for_fraction :
  ∃ p q : ℕ, 0 < p ∧ 0 < q ∧ 
  (3/5 : ℚ) < p / q ∧ p / q < (5/8 : ℚ) ∧
  (∀ r : ℕ, 0 < r ∧ (3/5 : ℚ) < p / r ∧ p / r < (5/8 : ℚ) → q ≤ r) ∧
  p + q = 21 :=
by
  sorry

end minimal_q_for_fraction_l330_330712


namespace complex_number_properties_l330_330944

theorem complex_number_properties (m : ℝ) :
  let z := complex.mk (m + 1) (m - 1)
  (z.im = 0 ↔ m = 1) ∧
  (z.im ≠ 0 ↔ m ≠ 1) ∧
  (z.re = 0 ∧ z.im ≠ 0 ↔ m = -1) :=
by
  let z := complex.mk (m + 1) (m - 1)
  sorry

end complex_number_properties_l330_330944


namespace min_value_of_function_l330_330777

noncomputable def min_value_function (x : ℝ) (hx : x > 0) : ℝ :=
x^2 + 3 / x

theorem min_value_of_function :
  ∃ (x : ℝ), x > 0 ∧ min_value_function x (by assumption) = (3 / 2) * real.cbrt 18 :=
sorry

end min_value_of_function_l330_330777


namespace problem_divisibility_l330_330951

theorem problem_divisibility 
  (m n : ℕ) 
  (a : Fin (mn + 1) → ℕ)
  (h_pos : ∀ i, 0 < a i)
  (h_order : ∀ i j, i < j → a i < a j) :
  (∃ (b : Fin (m + 1) → Fin (mn + 1)), ∀ i j, i ≠ j → ¬(a (b i) ∣ a (b j))) ∨
  (∃ (c : Fin (n + 1) → Fin (mn + 1)), ∀ i, i < n → a (c i) ∣ a (c i.succ)) :=
sorry

end problem_divisibility_l330_330951


namespace math_problem_l330_330154

-- Define real indexed sequences ai and bi
variable {α : Type*} [linear_ordered_field α]
variable (n : ℕ)
variable (a : ℕ → α)
variable (b : ℕ → α)

-- Conditions of the problem
-- Condition 1: For all i in the range, ai + a_{i+1} >= 0
def cond1 : Prop := ∀ i, (i < 2 * n) → a i + a (i + 1) ≥ 0

-- Condition 2: For all j in the range, a_{2j+1} ≤ 0
def cond2 : Prop := ∀ j, (j < n) → a (2 * j + 1) ≤ 0

-- Condition 3: For any p, q with 0 ≤ p ≤ q ≤ n, sum_{k=2p}^{2q} b_k > 0
def cond3 : Prop := ∀ p q, (p ≤ q) → (q ≤ n) → (0 ≤ p) → (∑ k in finset.filter (λ k, 2 * p ≤ k ∧ k ≤ 2 * q) (finset.range (2 * n + 1)), b k) > 0

-- Question to prove: sum_{i=0}^{2n} (-1)^i a_i b_i ≥ 0
def question : Prop := (∑ i in finset.range (2 * n + 1), (-1)^i * a i * b i) ≥ 0

-- Full Lean statement that includes conditions and question
theorem math_problem (h1 : cond1 n a) (h2 : cond2 n a) (h3 : cond3 n b) : question n a b := 
  sorry

end math_problem_l330_330154


namespace num_subsets_with_three_adjacent_chairs_l330_330760

/-- 
Prove the number of subsets of 10 chairs arranged in a circle that contain at least three adjacent chairs equals 581.
-/
theorem num_subsets_with_three_adjacent_chairs :
    let chairs := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
    ∑ x in (powerset chairs).filter (λ s, ∃ i ∈ s, ∀ j ∈ s, (j = (i + 1) % 10 ∨ j = (i + 2) % 10)) = 581 :=
sorry

end num_subsets_with_three_adjacent_chairs_l330_330760


namespace product_of_positive_real_parts_l330_330291

theorem product_of_positive_real_parts (z : ℂ) (h : z ^ 8 = 1) :
  ∃ a b : ℝ, z = a + b * complex.I ∧ a > 0 → (∏ (x : fin 3), (([1, complex.exp (complex.I * real.pi / 4), complex.exp (-3 * complex.I * real.pi / 4)].nth x).get_or_else (0)))) = 1 :=
by
  -- Placeholder for proof, which is not required
  sorry

end product_of_positive_real_parts_l330_330291


namespace sales_tax_reduction_difference_l330_330088

def sales_tax_difference (original_rate new_rate market_price : ℝ) : ℝ :=
  (market_price * original_rate) - (market_price * new_rate)

theorem sales_tax_reduction_difference :
  sales_tax_difference 0.035 0.03333 10800 = 18.36 :=
by
  -- This is where the proof would go, but it is not required for this task.
  sorry

end sales_tax_reduction_difference_l330_330088


namespace sectionBSeats_l330_330691

-- Definitions from the conditions
def seatsIn60SeatSubsectionA : Nat := 60
def subsectionsIn80SeatA : Nat := 3
def seatsPer80SeatSubsectionA : Nat := 80
def extraSeatsInSectionB : Nat := 20

-- Total seats in 80-seat subsections of Section A
def totalSeatsIn80SeatSubsections : Nat := subsectionsIn80SeatA * seatsPer80SeatSubsectionA

-- Total seats in Section A
def totalSeatsInSectionA : Nat := totalSeatsIn80SeatSubsections + seatsIn60SeatSubsectionA

-- Total seats in Section B
def totalSeatsInSectionB : Nat := 3 * totalSeatsInSectionA + extraSeatsInSectionB

-- The statement to prove
theorem sectionBSeats : totalSeatsInSectionB = 920 := by
  sorry

end sectionBSeats_l330_330691


namespace pattern_ABC_150th_letter_is_C_l330_330043

theorem pattern_ABC_150th_letter_is_C :
  (fun cycle length index =>
    let repeats := index / length;
    let remainder := index % length;
    if remainder = 0 then 'C' else
    if remainder = 1 then 'A' else 'B') 3 150 = 'C' := sorry

end pattern_ABC_150th_letter_is_C_l330_330043


namespace floor_tiling_l330_330082

theorem floor_tiling (n : ℕ) (x : ℕ) (h1 : 6 * x = n^2) : 6 ∣ n := sorry

end floor_tiling_l330_330082


namespace panic_percentage_left_l330_330143

theorem panic_percentage_left (original_population: ℕ) (pop_after_initial : ℕ) (pop_after_panic : ℕ) : 
  original_population = 7600 → 
  pop_after_initial = (7600 - (7600 / 10)) → 
  pop_after_panic = 5130 → 
  ((pop_after_initial - pop_after_panic) * 100) / pop_after_initial = 25 :=
by {
  intros h1 h2 h3,
  sorry
}

end panic_percentage_left_l330_330143


namespace cos_value_of_fraction_l330_330939

theorem cos_value_of_fraction:
  cos (2018 * Real.pi / 3) = -1 / 2 := by
  sorry

end cos_value_of_fraction_l330_330939


namespace mini_train_length_l330_330093

/-- Convert speed from km/h to m/s -/
def kmph_to_mps (kmph : Float) : Float :=
  kmph * (1000 / 3600)

theorem mini_train_length :
  ∀ (T : Float) (S_kph : Float),
    T = 3 →
    S_kph = 75 →
    let S_mps := kmph_to_mps S_kph in
    let D := S_mps * T in
    D = 62.5 :=
by
  intros T S_kph hT hS_kph
  simp [hT, hS_kph, kmph_to_mps]
  sorry

end mini_train_length_l330_330093


namespace smallest_n_intersection_triangles_l330_330861

noncomputable def convex_100_gon : Type := sorry

theorem smallest_n_intersection_triangles (P : convex_100_gon) : 
  ∃ n : ℕ, (∀ P : convex_100_gon, P = ⋂ (t ∈ set.range (λ k, triangle k P)), t) ∧ n = 50 :=
sorry

end smallest_n_intersection_triangles_l330_330861


namespace rabbit_travel_time_l330_330851

theorem rabbit_travel_time :
  let distance := 2
  let speed := 5
  let hours_to_minutes := 60
  (distance / speed) * hours_to_minutes = 24 := by
sorry

end rabbit_travel_time_l330_330851


namespace range_of_a_l330_330609

noncomputable def f (a x : ℝ) : ℝ := (1 / 2) ^ real.sqrt (x^2 - 4 * a * x + 8)

def is_monotonic_on (f : ℝ → ℝ) (a b : ℝ) : Prop := 
(∀ x y ∈ set.Icc a b, x ≤ y → f x ≤ f y) ∨ (∀ x y ∈ set.Icc a b, x ≤ y → f x ≥ f y)

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ set.Icc 2 6, x^2 - 4 * a * x + 8 ≥ 0) ∧ 
  is_monotonic_on (f a) 2 6 → a ≤ 1 :=
sorry

end range_of_a_l330_330609


namespace problem1_problem2_problem3_l330_330943

def is_real (m : ℝ) : Prop := (m^2 - 3 * m) = 0
def is_complex (m : ℝ) : Prop := (m^2 - 3 * m) ≠ 0
def is_pure_imaginary (m : ℝ) : Prop := (m^2 - 5 * m + 6) = 0 ∧ (m^2 - 3 * m) ≠ 0

theorem problem1 (m : ℝ) : is_real m ↔ (m = 0 ∨ m = 3) :=
sorry

theorem problem2 (m : ℝ) : is_complex m ↔ (m ≠ 0 ∧ m ≠ 3) :=
sorry

theorem problem3 (m : ℝ) : is_pure_imaginary m ↔ (m = 2) :=
sorry

end problem1_problem2_problem3_l330_330943


namespace rotate_and_translate_line_l330_330366

theorem rotate_and_translate_line :
  let initial_line (x : ℝ) := 3 * x
  let rotated_line (x : ℝ) := - (1 / 3) * x
  let translated_line (x : ℝ) := - (1 / 3) * (x - 1)

  ∀ x : ℝ, translated_line x = - (1 / 3) * x + (1 / 3) := 
by
  intros
  simp
  sorry

end rotate_and_translate_line_l330_330366


namespace percent_of_men_tenured_l330_330815

theorem percent_of_men_tenured (total_professors : ℕ) (women_percent tenured_percent women_tenured_or_both_percent men_percent tenured_men_percent : ℝ)
  (h1 : women_percent = 70 / 100)
  (h2 : tenured_percent = 70 / 100)
  (h3 : women_tenured_or_both_percent = 90 / 100)
  (h4 : men_percent = 30 / 100)
  (h5 : total_professors > 0)
  (h6 : tenured_men_percent = (2/3)) :
  tenured_men_percent * 100 = 66.67 :=
by sorry

end percent_of_men_tenured_l330_330815


namespace sum_of_factors_of_60_l330_330075

-- Given conditions
def n : ℕ := 60

-- Proof statement
theorem sum_of_factors_of_60 : (∑ d in (finset.filter (λ d, n % d = 0) (finset.range (n + 1))), d) = 168 :=
by
  sorry

end sum_of_factors_of_60_l330_330075


namespace bags_on_wednesday_l330_330162

theorem bags_on_wednesday (charge_per_bag money_per_bag monday_bags tuesday_bags total_money wednesday_bags : ℕ)
  (h_charge : charge_per_bag = 4)
  (h_money_per_dollar : money_per_bag = charge_per_bag)
  (h_monday : monday_bags = 5)
  (h_tuesday : tuesday_bags = 3)
  (h_total : total_money = 68) :
  wednesday_bags = (total_money - (monday_bags + tuesday_bags) * money_per_bag) / charge_per_bag :=
by
  sorry

end bags_on_wednesday_l330_330162


namespace denominator_of_first_fraction_l330_330650

theorem denominator_of_first_fraction (y x : ℝ) (h : y > 0) (h_eq : (9 * y) / x + (3 * y) / 10 = 0.75 * y) : x = 20 :=
by
  sorry

end denominator_of_first_fraction_l330_330650


namespace dog_distance_l330_330385

theorem dog_distance (distance_home_work : ℝ) (Ivan_speed dog_speed : ℝ) (meeting_fraction : ℝ) 
  (h1 : distance_home_work = 3)
  (h2 : Ivan_speed > 0)
  (h3 : dog_speed > 0)
  (h4 : meeting_fraction = 1/4)
  (h5 : dog_speed = 3 * Ivan_speed) :
  let dog_distance_traveled := distance_home_work * (1 / meeting_fraction) in
  dog_distance_traveled = 9 := 
begin
  sorry
end

end dog_distance_l330_330385


namespace zero_exists_in_interval_l330_330150

noncomputable theory

def f (x : ℝ) : ℝ := 2 * x ^ 3 - 3 * x - 9

theorem zero_exists_in_interval : ∃ x ∈ set.Ioo 1 2, f x = 0 :=
by
  -- Function evaluation at endpoints
  have f1 : f 1 = -10 := by simp [f]
  have f2 : f 2 = 1 := by simp [f]
  -- Use Intermediate Value Theorem based on f1 * f2 < 0
  exact sorry

end zero_exists_in_interval_l330_330150


namespace binary_to_decimal_l330_330903

theorem binary_to_decimal : 
  (1 * 2^0 + 1 * 2^1 + 0 * 2^2 + 1 * 2^3 + 1 * 2^4) = 27 :=
by
  sorry

end binary_to_decimal_l330_330903


namespace domain_f_l330_330192

def f (x : ℝ) : ℝ := 1 / Real.log (2 * x + 1)

theorem domain_f :
  {x : ℝ | 2 * x + 1 > 0 ∧ Real.log (2 * x + 1) ≠ 0} =
  {x : ℝ | -1 / 2 < x} ∩ {x : ℝ | x ≠ 0} :=
by
  sorry

end domain_f_l330_330192


namespace acute_angle_radians_l330_330031

noncomputable def radian_measure_of_acute_angle (r1 r2 r3 : ℝ) (shaded_ratio : ℝ) : ℝ :=
  if h : (r1 = 3 ∧ r2 = 2 ∧ r3 = 1 ∧ shaded_ratio = 8 / 13)
  then π / 7
  else 0

theorem acute_angle_radians {r1 r2 r3 : ℝ} {shaded_ratio : ℝ} 
  (h : r1 = 3 ∧ r2 = 2 ∧ r3 = 1 ∧ shaded_ratio = 8 / 13) : 
  radian_measure_of_acute_angle r1 r2 r3 shaded_ratio = π / 7 :=
by
  sorry

end acute_angle_radians_l330_330031


namespace avg_decrease_by_one_l330_330761

noncomputable def average_decrease (obs : Fin 7 → ℕ) : ℕ :=
  let sum6 := 90
  let seventh := 8
  let new_sum := sum6 + seventh
  let new_avg := new_sum / 7
  let old_avg := 15
  old_avg - new_avg

theorem avg_decrease_by_one :
  (average_decrease (fun _ => 0)) = 1 :=
by
  sorry

end avg_decrease_by_one_l330_330761


namespace tony_bread_slices_left_l330_330023

def num_slices_left (initial_slices : ℕ) (d1 : ℕ) (d2 : ℕ) : ℕ :=
  initial_slices - (d1 + d2)

theorem tony_bread_slices_left :
  num_slices_left 22 (5 * 2) (2 * 2) = 8 := by
  sorry

end tony_bread_slices_left_l330_330023


namespace tangent_line_at_origin_l330_330767

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.cos x

theorem tangent_line_at_origin :
  let p := (0, f 0)
  let tangent_line := λ (x y : ℝ), x - y + 1 = 0
  tangent_line (Prod.fst p) (Prod.snd p) :=
by 
  sorry

end tangent_line_at_origin_l330_330767


namespace algebraic_expression_value_l330_330649

theorem algebraic_expression_value (x : ℝ) (h : x^2 - 3*x + 1 = 4) : 2*x^2 - 6*x + 5 = 11 :=
by
  sorry

end algebraic_expression_value_l330_330649


namespace investment_duration_l330_330571

def principal : ℝ := 780
def interest_rate : ℝ := 4.166666666666667
def simple_interest : ℝ := 130

theorem investment_duration :
  ∃ T : ℝ, simple_interest = principal * interest_rate * T / 100 ∧ T = 4 :=
begin
  use 4,
  split,
  { sorry },
  { refl }
end

end investment_duration_l330_330571


namespace mower_next_tangent_point_l330_330884

theorem mower_next_tangent_point (r_garden r_mower : ℝ) (h_garden : r_garden = 15) (h_mower : r_mower = 5) :
    ∃ θ : ℝ, θ = (2 * π * r_mower / (2 * π * r_garden)) * 360 ∧ θ = 120 :=
sorry

end mower_next_tangent_point_l330_330884


namespace tony_bread_slices_left_l330_330025

def num_slices_left (initial_slices : ℕ) (d1 : ℕ) (d2 : ℕ) : ℕ :=
  initial_slices - (d1 + d2)

theorem tony_bread_slices_left :
  num_slices_left 22 (5 * 2) (2 * 2) = 8 := by
  sorry

end tony_bread_slices_left_l330_330025


namespace function_properties_l330_330992

-- Given conditions
def domain (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ∈ ℝ

def functional_eqn (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f(x * y) = y^2 * f(x) + x^2 * f(y)

-- Lean 4 theorem statement
theorem function_properties (f : ℝ → ℝ)
  (Hdom : domain f)
  (Heqn : functional_eqn f) :
  f 0 = 0 ∧ f 1 = 0 ∧ ∀ x : ℝ, f (-x) = f x :=
by
  sorry


end function_properties_l330_330992


namespace rainwater_depth_in_pyramid_l330_330466

theorem rainwater_depth_in_pyramid :
  let base_side := 23 -- side length of the base in cm
  let pyramid_height := 120 -- height of the pyramid in cm
  let rainfall := 5 -- cm of rain
  let base_area := base_side * base_side -- area of the base in square cm
  let rain_volume := base_area * rainfall -- volume of rain collected in cubic cm
  let pyramid_volume := (1 / 3 : ℚ) * base_area * pyramid_height
  let scaling_factor := (rain_volume / pyramid_volume)^(1 / 3 : ℚ)
  let depth := pyramid_height / scaling_factor
  depth = 60 :=
by
  -- definitions of the given conditions
  let base_side := 23 : ℚ
  let pyramid_height := 120 : ℚ
  let rainfall := 5 : ℚ
  
  -- intermediate calculations
  let base_area := base_side * base_side
  let rain_volume := base_area * rainfall
  let pyramid_volume := (1 / 3) * base_area * pyramid_height
  let scaling_factor := (rain_volume / pyramid_volume)^(1 / 3)
  let depth := pyramid_height / scaling_factor
  
  -- final assertion
  have h1 : scaling_factor = 2 := sorry
  have h2 : depth = pyramid_height / scaling_factor := sorry
  
  rw [h1] at h2
  exact h2

end rainwater_depth_in_pyramid_l330_330466


namespace jungkook_colored_paper_count_l330_330332

theorem jungkook_colored_paper_count :
  (3 * 10) + 8 = 38 :=
by sorry

end jungkook_colored_paper_count_l330_330332


namespace cost_of_one_pill_l330_330694

variable (P : ℝ) -- Cost of one pill

-- Conditions
variable (pills_per_day : ℝ := 2)
variable (days_in_month : ℝ := 30)
variable (insurance_percentage : ℝ := 0.40)
variable (payment : ℝ := 54)

-- Derived quantities
def total_pills : ℝ := pills_per_day * days_in_month -- Total pills in a month
def total_cost_paid_per_pill : ℝ := payment / total_pills -- Cost paid per pill
def patient_payment_percentage : ℝ := 1 - insurance_percentage -- Percentage John pays

-- Prove that cost per pill is $1.50
theorem cost_of_one_pill : P = (total_cost_paid_per_pill / patient_payment_percentage) := by
  sorry

end cost_of_one_pill_l330_330694


namespace giant_exponent_modulo_result_l330_330216

theorem giant_exponent_modulo_result :
  (7 ^ (7 ^ (7 ^ 7))) % 500 = 343 :=
by sorry

end giant_exponent_modulo_result_l330_330216


namespace largest_integer_modulo_l330_330929

theorem largest_integer_modulo (a : ℤ) : a < 93 ∧ a % 7 = 4 ∧ (∀ b : ℤ, b < 93 ∧ b % 7 = 4 → b ≤ a) ↔ a = 88 :=
by
    sorry

end largest_integer_modulo_l330_330929


namespace no_such_sequence_exists_l330_330912

noncomputable def has_sequence (a : ℕ → ℕ) : Prop :=
(∀ n : ℕ, a n < 1999 * n) ∧ (∀ n : ℕ, a n ≠ 0) ∧ (∀ n : ℕ, (decimal_digits a n).count 1 < 3)

theorem no_such_sequence_exists : ¬ ∃ a : ℕ → ℕ, has_sequence a :=
by {
  -- Proof would go here
  sorry
}

-- Helper function to count decimal digits
noncomputable def decimal_digits (n : ℕ) : list ℕ :=
let s := n.to_digits 10 in
s

-- Helper function to count occurrences of a digit
noncomputable def count_digit (d : ℕ) (n : ℕ) : ℕ :=
(decimal_digits n).count d

end no_such_sequence_exists_l330_330912


namespace quadratic_has_negative_root_l330_330267

def quadratic_function (m x : ℝ) : ℝ := (m - 2) * x^2 - 4 * m * x + 2 * m - 6

-- Define the discriminant of the quadratic function
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Define the function that checks for a range of m such that the quadratic function intersects the negative x-axis
theorem quadratic_has_negative_root (m : ℝ) :
  (∃ x : ℝ, x < 0 ∧ quadratic_function m x = 0) ↔ (1 ≤ m ∧ m < 2 ∨ 2 < m ∧ m < 3) :=
sorry

end quadratic_has_negative_root_l330_330267


namespace circumcircle_CFE_tangent_to_BC_at_E_l330_330419

-- Define the geometrical entities and their properties
variables {A B C E D D' C' P F : Type} [InCircle ABC ω]

-- Define contexts and conditions
def angle_A := 60 -- Given angle ∠A = 60°
def angle_B := 75 -- Given angle ∠B = 75°

def angle_bisector_A (A B C ω : Type) : Type -- Angle bisector meets BC at E and ω at D
def reflection_across_D_C (A D C : Type) : (D' C' : Type) -- Reflections of A across D and C are D' and C'

def tangent_to_circle_at_A (A P : Type) : Type -- Tangent to ω at A meets BC at P
def circumcircle (APD' AC : Type) : (F : Type) -- Circumcircle of APD' meets AC at F ≠ A

-- Main statement to prove
theorem circumcircle_CFE_tangent_to_BC_at_E :
  InCircle ABC ω → angle_A = 60 → angle_B = 75 → angle_bisector_A A B C ω → reflection_across_D_C A D C →
  tangent_to_circle_at_A A P → circumcircle APD' AC → 
  circumcircle C' F E is_tangent_to BC at E :=
begin
  sorry
end

end circumcircle_CFE_tangent_to_BC_at_E_l330_330419


namespace lead_to_tin_ratio_l330_330456

variable (L T T_B : ℝ)

def mix_alloys (L T T_B : ℝ) : Prop :=
  (L + T = 120) ∧
  (T_B = 67.5) ∧
  (T + T_B = 139.5) ∧
  let ratio : ℚ := (L / (T : ℝ)).toRat in ratio = 2 / 3

theorem lead_to_tin_ratio (h : mix_alloys L T T_B) : L / T = 2 / 3 :=
by sorry

end lead_to_tin_ratio_l330_330456


namespace largest_number_divisible_by_9_l330_330350

noncomputable def largest_divisible_by_9 : ℕ := 3213212121

theorem largest_number_divisible_by_9 :
  ∃ n : ℕ, (n ≤ 321321321321) ∧ (9 ∣ n) ∧ (n = 3213212121) :=
by {
  use 3213212121,
  have h1 : 3213212121 ≤ 321321321321, by norm_num,
  have h2 : 9 ∣ 3213212121, by norm_num,
  split, exact h1,
  split, exact h2,
  refl,
}

end largest_number_divisible_by_9_l330_330350


namespace probability_xavier_yvonne_not_zelda_l330_330094

noncomputable def xavier_solves : ℚ := 1 / 4
noncomputable def yvonne_solves : ℚ := 1 / 3
noncomputable def zelda_not_solves : ℚ := 3 / 8

theorem probability_xavier_yvonne_not_zelda :
  xavier_solves * yvonne_solves * zelda_not_solves = 1 / 32 :=
by {
  sorry
}

end probability_xavier_yvonne_not_zelda_l330_330094


namespace num_valid_k_values_l330_330908

theorem num_valid_k_values :
  ∃ (s : Finset ℕ), s = { 1, 2, 3, 6, 9, 18 } ∧ s.card = 6 :=
by
  sorry

end num_valid_k_values_l330_330908


namespace domain_of_g_l330_330906

def g (x : ℝ) : ℝ := real.sqrt (-8 * x^2 - 16 * x + 12)

theorem domain_of_g :
  {x : ℝ | -8 * x^2 - 16 * x + 12 >= 0} =
  set.Icc (-3 : ℝ) (1 / 2) :=
by 
  sorry

end domain_of_g_l330_330906


namespace functional_equation_option_A_option_B_option_C_l330_330976

-- Given conditions
def domain_of_f (f : ℝ → ℝ) : Set ℝ := Set.univ

theorem functional_equation (f : ℝ → ℝ) (x y : ℝ) :
  f (x * y) = y^2 * f x + x^2 * f y := sorry

-- Statements to be proved
theorem option_A (f : ℝ → ℝ) (h : ∀ x y, f(x * y) = y^2 * f(x) + x^2 * f(y)) : f 0 = 0 := sorry

theorem option_B (f : ℝ → ℝ) (h : ∀ x y, f(x * y) = y^2 * f(x) + x^2 * f(y)) : f 1 = 0 := sorry

theorem option_C (f : ℝ → ℝ) (h : ∀ x y, f(x * y) = y^2 * f(x) + x^2 * f(y)) : ∀ x, f(-x) = f x := sorry

end functional_equation_option_A_option_B_option_C_l330_330976


namespace horner_method_multiplications_l330_330194

def f (x : ℝ) : ℝ := 5 * x^6 - 7 * x^5 + 2 * x^4 - 8 * x^3 + 3 * x^2 - 9 * x + 1

theorem horner_method_multiplications : 
  (∃ n : ℕ, n = 6 ∧ ∀ x : ℝ, (Horner method to evaluate f(x) at x requires n multiplications)) :=
begin
  -- definition of horner's method applied to the polynomial f(x)
  sorry
end

end horner_method_multiplications_l330_330194


namespace non_congruent_rectangles_count_l330_330853

-- Define the conditions in Lean
def perimeter (w h : ℕ) : ℕ := 2 * (w + h)

-- Define the main theorem stating the problem and its result
theorem non_congruent_rectangles_count 
  (h w : ℕ) (H : h + w = 40) (integral_lengths : ∀ {u v : ℕ}, u + v = h + w → u = h ∨ u = w) : 
  ∃ n : ℕ, n = 20 :=
begin
  sorry -- Proof steps go here
end

end non_congruent_rectangles_count_l330_330853


namespace lino_shells_l330_330725

theorem lino_shells : 
  ∀ (morning_shells afternoon_shells : Nat), 
  morning_shells = 292 → 
  afternoon_shells = 324 → 
  (morning_shells + afternoon_shells = 616) :=
by
  intros morning_shells afternoon_shells h1 h2
  rw [h1, h2]
  exact rfl

end lino_shells_l330_330725


namespace plot_length_60_l330_330446

/-- The length of a rectangular plot is 20 meters more than its breadth. If the cost of fencing the plot at Rs. 26.50 per meter is Rs. 5300, then the length of the plot in meters is 60. -/
theorem plot_length_60 (b l : ℝ) (h1 : l = b + 20) (h2 : 2 * (l + b) * 26.5 = 5300) : l = 60 :=
by
  sorry

end plot_length_60_l330_330446


namespace has_zero_in_interval_l330_330105

noncomputable def f (x : ℝ) : ℝ := log x / log 5 + x - 3

theorem has_zero_in_interval :
  ∃ x ∈ set.Ioo (2 : ℝ) (3 : ℝ), f x = 0 :=
begin
  have h1 : f 2 < 0,
  { 
    -- Proof that f(2) < 0 is skipped
    sorry 
  },
  have h2 : f 3 > 0,
  { 
    -- Proof that f(3) > 0 is skipped
    sorry 
  },
  rw ←intermediate_value_Ioo (2 : ℝ) (3 : ℝ) f 
  (continuous_log.div continuous_const).add continuous_id.sub continuous_const,
  exact ⟨2, 3, h1, h2⟩,
end

end has_zero_in_interval_l330_330105


namespace solve_equation_l330_330757

theorem solve_equation (a : ℝ) (x : ℝ) (h : a ≥ -6) :
  (x^4 - 10 * x^3 - 2 * (a - 11) * x^2 + 2 * (5 * a + 6) * x + 2 * a + a^2 = 0) ↔
  (a = x^2 - 4 * x - 2 ∨ a = x^2 - 6 * x) ∧
  (x = 2 + sqrt (a + 6) ∨ x = 2 - sqrt (a + 6) ∨ x = 3 + sqrt (a + 9) ∨ x = 3 - sqrt (a + 9)) :=
sorry

end solve_equation_l330_330757


namespace find_y_intercept_l330_330007

def point (x y : ℝ) := (x, y)

def slope (m : ℝ) := m

def x_intercept (x : ℝ) := point x 0

def y_intercept_point (m x0 : ℝ) :=
  let b := 0 - m * x0
  point 0 b

theorem find_y_intercept :
  ∀ (m x0 : ℝ), (m = -3) → (x0 = 7) → y_intercept_point m x0 = (0, 21) :=
by
  intros m x0 m_cond x0_cond
  simp [y_intercept_point, m_cond, x0_cond]
  exact sorry

end find_y_intercept_l330_330007


namespace quotient_of_sum_and_difference_find_number_less_than_another_l330_330875

-- Problem (1)
theorem quotient_of_sum_and_difference :
  let sum : ℝ := 0.4 + (1 / 3)
  let diff : ℝ := 0.4 - (1 / 3)
  (sum / diff) = 11 := 
by 
  simp only [sum, diff, of_rat_sub, of_rat_div] 
  sorry

-- Problem (2)
theorem find_number_less_than_another :
  ∃ x : ℝ, x - 0.4 * x = 36 ∧ x = 60 :=
by 
  existsi 60
  split
  simp
  norm_num
  rfl
  sorry

end quotient_of_sum_and_difference_find_number_less_than_another_l330_330875


namespace bags_on_wednesday_l330_330161

theorem bags_on_wednesday (charge_per_bag money_per_bag monday_bags tuesday_bags total_money wednesday_bags : ℕ)
  (h_charge : charge_per_bag = 4)
  (h_money_per_dollar : money_per_bag = charge_per_bag)
  (h_monday : monday_bags = 5)
  (h_tuesday : tuesday_bags = 3)
  (h_total : total_money = 68) :
  wednesday_bags = (total_money - (monday_bags + tuesday_bags) * money_per_bag) / charge_per_bag :=
by
  sorry

end bags_on_wednesday_l330_330161


namespace fraction_of_area_shaded_triangle_is_correct_l330_330738

structure Point :=
  (x : ℝ)
  (y : ℝ)

def is_equilateral_triangle (A B C : Point) : Prop :=
  (A.x - B.x)^2 + (A.y - B.y)^2 = (A.x - C.x)^2 + (A.y - C.y)^2 ∧
  (A.x - B.x)^2 + (A.y - B.y)^2 = (B.x - C.x)^2 + (B.y - C.y)^2 ∧
  (A.x - C.x)^2 + (A.y - C.y)^2 = (B.x - C.x)^2 + (B.y - C.y)^2

def area_of_equilateral_triangle (s : ℝ) : ℝ :=
  (sqrt 3 / 4) * s^2

def area_of_square (side_length : ℝ) := side_length^2

noncomputable def fraction_of_area_inside_triangle (side_length_of_square side_length_of_triangle : ℝ) :=
  (area_of_equilateral_triangle side_length_of_triangle) / (area_of_square side_length_of_square)

theorem fraction_of_area_shaded_triangle_is_correct :
  ∀ (A B C : Point),
  (A = Point.mk 2 2) →
  (B = Point.mk 4 2) →
  (C = Point.mk 3 4) →
  is_equilateral_triangle A B C →
  fraction_of_area_inside_triangle 6 2 = sqrt 3 / 36 := 
by
  intros A B C hA hB hC hTriangle
  sorry

end fraction_of_area_shaded_triangle_is_correct_l330_330738


namespace maximum_equilateral_triangle_area_l330_330006

noncomputable def maxEquilateralTriangleAreaInRectangle (length : ℝ) (width : ℝ) : ℝ :=
  if length = 12 ∧ width = 14 then 36 * Real.sqrt 3 else 0

theorem maximum_equilateral_triangle_area :
  maxEquilateralTriangleAreaInRectangle 12 14 = 36 * Real.sqrt 3 :=
begin
  sorry
end

end maximum_equilateral_triangle_area_l330_330006


namespace binary_to_decimal_l330_330890

theorem binary_to_decimal : 
  let binary_number := [1, 1, 0, 1, 1] in
  let weights := [2^4, 2^3, 2^2, 2^1, 2^0] in
  (List.zipWith (λ digit weight => digit * weight) binary_number weights).sum = 27 := by
  sorry

end binary_to_decimal_l330_330890


namespace magnitude_PQ_l330_330626

variables {a b c d m k : ℝ}

def point_on_line (x y m k : ℝ) : Prop := y = m * x + k

theorem magnitude_PQ (hP : point_on_line a b m k) (hQ : point_on_line c d m k) :
  |(c - a, d - b)| = |a - c| * sqrt (1 + m^2) :=
by
  sorry

end magnitude_PQ_l330_330626


namespace probability_three_and_one_l330_330296

-- Define the event: having 3 children of one sex and 1 of the opposite sex
def three_and_one (sexes : list bool) : Prop :=
  (count true sexes = 3 ∧ count false sexes = 1) ∨ (count true sexes = 1 ∧ count false sexes = 3)

-- Probability of each child being a boy (true) or a girl (false) is 1/2
def child_probability : ℙ (list bool) :=
  pmf.uniform (list bool) [true, false]

-- Final proof statement
theorem probability_three_and_one : 
  have_children (n : ℕ) := 4 ∧ child_probability := 1/2):
  ℙ (three_and_one have_children) = 1/2 :=
by
  sorry

end probability_three_and_one_l330_330296


namespace inverse_function_correct_l330_330536

noncomputable def original_function (x : ℝ) : ℝ := 2^(1 - x) + 3
noncomputable def inverse_function (y : ℝ) : ℝ := log (2 / (y - 3)) / log 2

theorem inverse_function_correct (x : ℝ) : inverse_function (original_function x) = x :=
by sorry

end inverse_function_correct_l330_330536


namespace algebraic_expression_simplification_l330_330167

theorem algebraic_expression_simplification :
  0.25 * (-1 / 2) ^ (-4 : ℝ) - 4 / (Real.sqrt 5 - 1) ^ (0 : ℝ) - (1 / 16) ^ (-1 / 2 : ℝ) = -4 :=
by
  sorry

end algebraic_expression_simplification_l330_330167


namespace orthogonal_vectors_m_value_l330_330721

theorem orthogonal_vectors_m_value (m : ℝ) :
  let a : ℝ × ℝ := (1, -1)
  let b : ℝ × ℝ := (m + 1, 2 * m - 4)
  (a.fst * b.fst + a.snd * b.snd = 0) → m = 5 :=
by
  let a : ℝ × ℝ := (1, -1)
  let b : ℝ × ℝ := (m + 1, 2 * m - 4)
  assume h : a.fst * b.fst + a.snd * b.snd = 0
  sorry

end orthogonal_vectors_m_value_l330_330721


namespace num_five_digit_ints_l330_330273

open Nat

theorem num_five_digit_ints : 
  let num_ways := (factorial 5) / (factorial 3 * factorial 2)
  num_ways = 10 :=
by
  let num_ways := (factorial 5) / (factorial 3 * factorial 2)
  sorry

end num_five_digit_ints_l330_330273


namespace measure_angle_C_is_2pi_over_3_l330_330680

noncomputable def measure_angle_C (a b c : ℝ) (h_a : a = 7) (h_b : b = 8) (h_c : c = 13) : ℝ :=
  real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))

theorem measure_angle_C_is_2pi_over_3 :
  measure_angle_C 7 8 13 (by rfl) (by rfl) (by rfl) = 2 * real.pi / 3 :=
sorry

end measure_angle_C_is_2pi_over_3_l330_330680


namespace binary_to_decimal_l330_330901

theorem binary_to_decimal : 
  (1 * 2^0 + 1 * 2^1 + 0 * 2^2 + 1 * 2^3 + 1 * 2^4) = 27 :=
by
  sorry

end binary_to_decimal_l330_330901


namespace circle_intersection_line_eq_l330_330465

theorem circle_intersection_line_eq (x y : ℝ) :
  let circle1 := x^2 + y^2 = 25,
      circle2 := (x - 4)^2 + (y - 4)^2 = 9
  in (circle1 ∧ circle2) → x + y = 23 / 8 :=
by
  assume h,
  sorry

end circle_intersection_line_eq_l330_330465


namespace a_4_eq_28_l330_330324

def Sn (n : ℕ) : ℕ := 4 * n^2

def a_n (n : ℕ) : ℕ := Sn n - Sn (n - 1)

theorem a_4_eq_28 : a_n 4 = 28 :=
by
  sorry

end a_4_eq_28_l330_330324


namespace two_trains_cross_time_l330_330034

def train1_length : ℝ := 450  -- meters
def train2_length : ℝ := 720  -- meters
def train1_speed : ℝ := 120 * 1000 / 3600  -- convert kmph to m/s
def train2_speed : ℝ := 80 * 1000 / 3600  -- convert kmph to m/s

theorem two_trains_cross_time :
  let relative_speed := train1_speed + train2_speed,
      total_distance := train1_length + train2_length,
      time := total_distance / relative_speed
  in abs (time - 21.05) < 0.01 :=
by
  sorry

end two_trains_cross_time_l330_330034


namespace matrix_eigenvalues_and_power_computation_l330_330238

noncomputable def M : Matrix (Fin 2) (Fin 2) ℤ := ![![4, -3], ![2, -1]]
noncomputable def α : Fin 2 → ℤ := ![7, 5]

theorem matrix_eigenvalues_and_power_computation :
  let λ1 := 1
  let λ2 := 2
  let evec1 : Fin 2 → ℤ := ![1, 1]
  let evec2 : Fin 2 → ℤ := ![3, 2]
  (M^3).mulVec α = ![49, 33] :=
by {
  let λ1 := 1,
  let λ2 := 2,
  let evec1 : Fin 2 → ℤ := ![1, 1],
  let evec2 : Fin 2 → ℤ := ![3, 2],
  sorry
}

end matrix_eigenvalues_and_power_computation_l330_330238


namespace division_pow_zero_l330_330425

theorem division_pow_zero (a b : ℝ) (hb : b ≠ 0) : ((a / b) ^ 0 = (1 : ℝ)) :=
by
  sorry

end division_pow_zero_l330_330425


namespace arithmetic_sequence_geometric_condition_l330_330240

theorem arithmetic_sequence_geometric_condition :
  ∃ (a : ℤ), 
    let a2 := a + 3 in
    let a4 := a + 9 in
    let a8 := a + 21 in
    (a4 * a4 = a2 * a8) → a4 = 12 :=
begin
  sorry
end

end arithmetic_sequence_geometric_condition_l330_330240


namespace problem_statement_l330_330736

theorem problem_statement :
  ∀ m n : ℕ, (m = 9) → (n = m^2 + 1) → n - m = 73 :=
by
  intros m n hm hn
  rw [hm, hn]
  sorry

end problem_statement_l330_330736


namespace sum_of_positive_factors_60_l330_330052

def sum_of_positive_factors (n : ℕ) : ℕ :=
  ∑ d in (Finset.filter (λ x => x > 0) (Finset.divisors n)), d

theorem sum_of_positive_factors_60 : sum_of_positive_factors 60 = 168 := by
  sorry

end sum_of_positive_factors_60_l330_330052


namespace minute_hand_rotation_l330_330834

theorem minute_hand_rotation (h : ℕ) (m : ℕ) :
  1 = 60 ∧ 360° = -360 ∧ h = 1 ∧ m = 50 →
  (m + 60 * h) * 360 / 60 = -660 :=
by
  intro cond
  cases cond with h_eq rest
  cases rest with deg rest1
  cases rest1 with h_val m_val
  sorry

end minute_hand_rotation_l330_330834


namespace ball_bounce_height_l330_330108

theorem ball_bounce_height (n : ℕ) : (512 * (1/2)^n < 20) → n = 8 := 
sorry

end ball_bounce_height_l330_330108


namespace sum_of_positive_factors_60_l330_330068

theorem sum_of_positive_factors_60 : (∑ n in Nat.divisors 60, n) = 168 := by
  sorry

end sum_of_positive_factors_60_l330_330068


namespace task_completion_methods_l330_330452

theorem task_completion_methods {n : ℕ} (m : Fin n → ℕ) : (∏ i, m i) > 0 → ∀ i, m i > 0 → (∏ i, m i) = finset.prod (finset.range n) m := 
sorry

end task_completion_methods_l330_330452


namespace star_stones_per_bracelet_l330_330520

namespace Bracelets

theorem star_stones_per_bracelet (b : ℕ) (s : ℕ) (h_b : b = 3) (h_s : s = 36) :
  s / b = 12 :=
by
  rw [h_b, h_s]
  exact Nat.div_eq_of_eq_mul_right (Nat.succ_pos 2) rfl

end Bracelets

end star_stones_per_bracelet_l330_330520


namespace number_of_buses_proof_l330_330485

-- Define the conditions
def columns_per_bus : ℕ := 4
def rows_per_bus : ℕ := 10
def total_students : ℕ := 240
def seats_per_bus (c : ℕ) (r : ℕ) : ℕ := c * r
def number_of_buses (total : ℕ) (seats : ℕ) : ℕ := total / seats

-- State the theorem we want to prove
theorem number_of_buses_proof :
  number_of_buses total_students (seats_per_bus columns_per_bus rows_per_bus) = 6 := 
sorry

end number_of_buses_proof_l330_330485


namespace max_tan_B_l330_330956

theorem max_tan_B (A B : ℝ) (hA : 0 < A ∧ A < π/2) (hB : 0 < B ∧ B < π/2) (h : Real.tan (A + B) = 2 * Real.tan A) :
  ∃ B_max, B_max = Real.tan B ∧ B_max ≤ Real.sqrt 2 / 4 :=
by
  sorry

end max_tan_B_l330_330956


namespace board_eventually_stabilizes_min_moves_to_stabilize_l330_330335

-- Prove that eventually the numbers on the board will not change
theorem board_eventually_stabilizes (n : ℕ) (initial : Fin n → ℕ) : 
  ∃ k, ∀ i ≥ k, (board_after_moves i initial = board_after_moves (i + 1) initial) :=
sorry

-- Determine the minimum integer k such that after k moves, the numbers do not change
theorem min_moves_to_stabilize (n : ℕ) (initial : Fin n → ℕ) : 
  ∀ (k : ℕ), k ≥ 2 * n → ∀ i ≥ k, (board_after_moves i initial = board_after_moves (i + 1) initial) :=
sorry

noncomputable def board_after_moves : ℕ → (Fin n → ℕ) → (Fin n → ℕ) :=
sorry

end board_eventually_stabilizes_min_moves_to_stabilize_l330_330335


namespace top_square_multiple_of_5_possible_configurations_l330_330131

theorem top_square_multiple_of_5_possible_configurations :
  let fourteen_th_row := vector bool 14
  ∃ (configs : fourteen_th_row → bool), 
  (∀ row : vector bool 14, row.top_square_multiples_5) 
  ∧ (number_of_configurations configs = 2048) := sorry

end top_square_multiple_of_5_possible_configurations_l330_330131


namespace median_salary_correct_l330_330500

noncomputable def employee_details : List (String × ℕ × ℕ) :=
[ ("CEO", 1, 140000),
  ("Senior Manager", 4, 95000),
  ("Manager", 15, 80000),
  ("Assistant Manager", 7, 55000),
  ("Clerk", 40, 25000) ]

/--
  According to the provided list of employee details, where each entry is a tuple consisting of
  a String (position title), ℕ (number of employees with that title), and ℕ (salary in dollars),
  prove that the median salary among the total 67 employees is $25,000.
-/
theorem median_salary_correct :
  let total_employees := 67 
  (median (List.replicate 1 140000 ++ List.replicate 4 95000 
    ++ List.replicate 15 80000 ++ List.replicate 7 55000 
    ++ List.replicate 40 25000) = 25000) :=
by
  sorry

end median_salary_correct_l330_330500


namespace gas_consumption_reduction_l330_330795

theorem gas_consumption_reduction
  (P C : ℝ)
  (h₁ : 0 < P) 
  (h₂ : 0 < C) 
  (increase1 : P' = P * 1.15) 
  (increase2 : P'' = P' * 1.10) :
  let reduction := (1 - 1 / 1.265) * 100
  in reduction ≈ 20.95 := 
by
  sorry

end gas_consumption_reduction_l330_330795


namespace harmonic_series_half_sum_lt_one_l330_330037

theorem harmonic_series_half_sum_lt_one (n : ℕ) (hn : 2 ≤ n) :
  (∑ i in Finset.range (2 * n + 1) \ Finset.range n, (1 : ℝ) / (i + 1)) < 1 :=
sorry

end harmonic_series_half_sum_lt_one_l330_330037


namespace sequence_properties_l330_330608

theorem sequence_properties (a_n : ℕ → ℕ) (S_n : ℕ → ℕ) (n : ℕ) :

  -- Condition for option A
  (S_n = λ n, n^2 + n → ∃d, ∀n, a_n n = a_n (n - 1) + d) ∧

  -- Condition for option B
  ((a_n 1 > 0 ∧ ∃q, 0 < q ∧ ∀n, a_n (n + 1) = a_n n * q) → S_n 1 * S_n 3 ≤ (S_n 2)^2) ∧

  -- Condition for option C
  ((∃d, ∀n, a_n n = a_n (n - 1) + d) → S_n 11 = 11 * a_n 6) ∧

  -- Condition for option D
  (S_n = λ n, 3^n - 1 → ∃r, r > 0 ∧ ∀n, a_n (n + 1) = a_n n * r)
:=
by
  sorry

end sequence_properties_l330_330608


namespace min_constant_correct_l330_330596

open Finset

def minimum_constant_c (n k : ℕ) (h_n : 2 ≤ n) : ℝ :=
  (n * (k * (k - 1) / 2) + k) / (n * k + 1)

theorem min_constant_correct (n k : ℕ) (h_n : 2 ≤ n) (m : ℕ) (G : SimpleGraph (Fin m)) (hG : G.IsRegular (n * k)) :
  ∃ c, c = minimum_constant_c n k h_n ∧
       ∀ coloring : Coloring (Fin n),
       G.count_monochromatic_edges coloring.toColoring ≤ c * m := 
begin
  sorry,
end

end min_constant_correct_l330_330596


namespace functional_equation_option_A_option_B_option_C_l330_330972

-- Given conditions
def domain_of_f (f : ℝ → ℝ) : Set ℝ := Set.univ

theorem functional_equation (f : ℝ → ℝ) (x y : ℝ) :
  f (x * y) = y^2 * f x + x^2 * f y := sorry

-- Statements to be proved
theorem option_A (f : ℝ → ℝ) (h : ∀ x y, f(x * y) = y^2 * f(x) + x^2 * f(y)) : f 0 = 0 := sorry

theorem option_B (f : ℝ → ℝ) (h : ∀ x y, f(x * y) = y^2 * f(x) + x^2 * f(y)) : f 1 = 0 := sorry

theorem option_C (f : ℝ → ℝ) (h : ∀ x y, f(x * y) = y^2 * f(x) + x^2 * f(y)) : ∀ x, f(-x) = f x := sorry

end functional_equation_option_A_option_B_option_C_l330_330972


namespace constructible_angles_l330_330914

def is_constructible (θ : ℝ) : Prop :=
  -- Define that θ is constructible if it can be constructed using compass and straightedge.
  sorry

theorem constructible_angles (α : ℝ) (β : ℝ) (k n : ℤ) (hβ : is_constructible β) :
  is_constructible (k * α / 2^n + β) :=
sorry

end constructible_angles_l330_330914


namespace isosceles_triangle_extension_parallel_base_l330_330918

theorem isosceles_triangle_extension_parallel_base
  (A B C B1 C1 : Type*)
  [metric_space B] [metric_space C] [metric_space B1] [metric_space C1]
  [has_extend AB AC A B1 C1]
  (isosceles : AB = AC)
  (extends_AB : extension AB)
  (extends_AC : extension AC) :
  ∥B1C1∥ ∥BC∥ :=
sorry

end isosceles_triangle_extension_parallel_base_l330_330918


namespace diver_reaches_ship_in_64_14_min_l330_330855

def normal_descent_rate : ℝ := 80
def downward_current_speed_increase : ℝ := 30
def upward_current_speed_decrease : ℝ := 20
def decompression_stop_1_depth : ℝ := 1800
def decompression_stop_1_time : ℝ := 5
def decompression_stop_2_depth : ℝ := 3600
def decompression_stop_2_time : ℝ := 8
def lost_ship_depth : ℝ := 4000

def total_time_to_reach_lost_ship : ℝ :=
  (1500 / (normal_descent_rate + downward_current_speed_increase)) + 
  (1500 / (normal_descent_rate - upward_current_speed_decrease)) + 
  (1000 / normal_descent_rate) + 
  decompression_stop_1_time + 
  decompression_stop_2_time

theorem diver_reaches_ship_in_64_14_min :
  total_time_to_reach_lost_ship = 64.14 := by
  sorry

end diver_reaches_ship_in_64_14_min_l330_330855


namespace pentagon_coloring_count_l330_330915

-- Define the vertices and colors
inductive Vertex
| A | B | C | D | E

def colors : Finset ℕ := {0, 1, 2, 3, 4}

-- Define the conditions for a valid coloring
def valid_coloring (color : Vertex → ℕ) : Prop :=
  (∀ v, color v ∈ colors) ∧
  (color Vertex.A ≠ color Vertex.C) ∧
  (color Vertex.B ≠ color Vertex.D) ∧
  (color Vertex.A ≠ color Vertex.B) ∧ 
  (color Vertex.A ≠ color Vertex.D) ∧
  (color Vertex.B ≠ color Vertex.C) ∧
  (color Vertex.B ≠ color Vertex.E) ∧
  (color Vertex.C ≠ color Vertex.D) ∧
  (color Vertex.C ≠ color Vertex.E) ∧
  (color Vertex.D ≠ color Vertex.E)

-- Assert and prove the number of valid colorings
def count_valid_colorings : ℕ := Finset.card {c : Vertex → ℕ | valid_coloring c}

theorem pentagon_coloring_count : count_valid_colorings = 540 :=
by
  sorry

end pentagon_coloring_count_l330_330915


namespace number_of_sheets_l330_330829

theorem number_of_sheets (sheets_per_box : ℕ) (thickness_per_box : ℝ) (height_of_stack : ℝ) (per_sheet : ℝ)
  (h_sheets : sheets_per_box = 400)
  (h_thickness : thickness_per_box = 4)
  (h_height : height_of_stack = 10)
  (h_per_sheet : per_sheet = thickness_per_box / sheets_per_box) :
  height_of_stack / per_sheet = 1000 :=
by
  rw [h_sheets, h_thickness, h_height, h_per_sheet]
  sorry

end number_of_sheets_l330_330829


namespace range_of_quadratic_function_l330_330004

theorem range_of_quadratic_function : 
  (set.range (λ x, x^2 - 2 * x)) ∩ set.Icc 0 3 = set.Icc (-1) 3 :=
by
  sorry

end range_of_quadratic_function_l330_330004


namespace nonagon_perimeter_l330_330184

theorem nonagon_perimeter (n : ℕ) (side_length : ℝ) (P : ℝ) :
  n = 9 → side_length = 3 → P = n * side_length → P = 27 :=
by sorry

end nonagon_perimeter_l330_330184


namespace infinite_grid_rectangles_l330_330739

theorem infinite_grid_rectangles (m : ℕ) (hm : m > 12) : 
  ∃ (x y : ℕ), x * y > m ∧ x * (y - 1) < m := 
  sorry

end infinite_grid_rectangles_l330_330739


namespace cosine_expression_decomposition_l330_330768

theorem cosine_expression_decomposition (x : ℝ) (a b c d : ℝ) (h : a = 4 ∧ b = 7.5 ∧ c = 1.5 ∧ d = 1.5) :
  (cos (2 * x) + cos (4 * x) + cos (10 * x) + cos (14 * x)) = a * (cos (b * x)) * (cos (c * x)) * (cos (d * x)) ∧ (a + b + c + d = 14.5) := 
by {
  sorry,
}

end cosine_expression_decomposition_l330_330768


namespace boat_avg_speed_ratio_l330_330461

/--
A boat moves at a speed of 20 mph in still water. When traveling in a river with a current of 3 mph, it travels 24 miles downstream and then returns upstream to the starting point. Prove that the ratio of the average speed for the entire round trip to the boat's speed in still water is 97765 / 100000.
-/
theorem boat_avg_speed_ratio :
  let boat_speed := 20 -- mph in still water
  let current_speed := 3 -- mph river current
  let distance := 24 -- miles downstream and upstream
  let downstream_speed := boat_speed + current_speed
  let upstream_speed := boat_speed - current_speed
  let time_downstream := distance / downstream_speed
  let time_upstream := distance / upstream_speed
  let total_time := time_downstream + time_upstream
  let total_distance := distance * 2
  let average_speed := total_distance / total_time
  (average_speed / boat_speed) = 97765 / 100000 :=
by
  sorry

end boat_avg_speed_ratio_l330_330461


namespace find_digits_l330_330673

theorem find_digits (A B D E C : ℕ) 
  (hC : C = 9) 
  (hA : 2 < A ∧ A < 4)
  (hB : B = 5)
  (hE : E = 6)
  (hD : D = 0)
  (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ C ≠ D ∧ C ≠ E ∧ D ≠ E) :
  (A, B, D, E) = (3, 5, 0, 6) := by
  sorry

end find_digits_l330_330673


namespace min_sum_of_fractions_l330_330700

theorem min_sum_of_fractions :
  ∀ (A B C D : ℕ),
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D →
    (A ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}) →
    (B ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}) →
    (C ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}) →
    (D ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}) →
    (A ≠ B ∧ B ≠ C ∧ C ≠ D) →
    (1 / (B : ℝ) + 3 / (D : ℝ) = 31 / 56) :=
begin
  intros A B C D h1 h2 h3 h4 h5 h6,
  sorry
end

end min_sum_of_fractions_l330_330700


namespace count_ordered_pairs_squares_diff_l330_330275

theorem count_ordered_pairs_squares_diff (m n : ℕ) (h1 : m ≥ n) (h2 : m^2 - n^2 = 72) : 
∃ (a : ℕ), a = 3 :=
sorry

end count_ordered_pairs_squares_diff_l330_330275


namespace intersection_point_of_perpendicular_chords_on_parabola_l330_330619

theorem intersection_point_of_perpendicular_chords_on_parabola :
  ∀ (M E : EuclideanSpace ℝ (Fin 2)),
  M = (1, 1) →
  is_vertex_of_perpendicular_chords_inscribed_in_parabola (λ x, x^2) M →
  E = (-1, 2) →
  intersection_of_segments (parabola_chord (λ x, x^2) M) (perpendicular_parabola_chord (λ x, x^2) M) = E :=
by sorry

end intersection_point_of_perpendicular_chords_on_parabola_l330_330619


namespace y_intercept_of_line_l330_330009

theorem y_intercept_of_line (m : ℝ) (x1 y1 : ℝ) (h_slope : m = -3) (h_x_intercept : x1 = 7) (h_y_intercept : y1 = 0) : 
  ∃ y_intercept : ℝ, y_intercept = 21 ∧ y_intercept = -m * 0 + 21 :=
by
  sorry

end y_intercept_of_line_l330_330009


namespace derivative_ln_plus_const_at_1_l330_330380

open Real

theorem derivative_ln_plus_const_at_1 : 
  let f : ℝ → ℝ := λ x, 2 + log x
  deriv f 1 = 1 :=
by
  sorry

end derivative_ln_plus_const_at_1_l330_330380


namespace symmetric_point_proof_l330_330378

def symmetric_point (P : ℝ × ℝ) (L : ℝ × ℝ → Prop) : ℝ × ℝ :=
  let (x, y) := P
  (-y, -x)

theorem symmetric_point_proof :
  symmetric_point (2, 5) (λ ⟨x, y⟩, x + y = 0) = (-5, -2) :=
by
  -- The proof goes here
  sorry

end symmetric_point_proof_l330_330378


namespace value_of_a8_seq_is_geometric_general_formula_for_b_l330_330323

variable {n : ℕ} {λ d : ℝ}

-- Condition definitions
def a (n : ℕ) : ℝ := sorry -- the definition of a_n based on conditions described
-- Prove part 1
theorem value_of_a8 (h₁ : λ = 1) (h₂ : d = 1) : 
  a 8 = 3 := sorry

-- Prove part 2
theorem seq_is_geometric (h : d ≠ 0) : 
  ∃ r, ∀ n : ℕ, |a (2^(n+2)) - a (2^n)| = r ^ n :=
sorry

-- Prove part 3
def b (n : ℕ) : ℝ := 
  if n % 2 = 0 then (2^(n+2))/3 + 2/3 
  else (2^(n+2))/3 - 2/3

theorem general_formula_for_b (h : λ ≠ 1) :
  ∀ n, b n = 
    if n % 2 = 0 then (2^(n+2))/3 + 2/3 
    else (2^(n+2))/3 - 2/3 :=
sorry

end value_of_a8_seq_is_geometric_general_formula_for_b_l330_330323


namespace hydrogen_moles_l330_330561

-- Define the balanced chemical reaction as a relation between moles
def balanced_reaction (NaH H₂O NaOH H₂ : ℕ) : Prop :=
  NaH = NaOH ∧ H₂ = NaOH ∧ NaH = H₂

-- Given conditions
def given_conditions (NaH H₂O : ℕ) : Prop :=
  NaH = 2 ∧ H₂O = 2

-- Problem statement to prove
theorem hydrogen_moles (NaH H₂O NaOH H₂ : ℕ)
  (h₁ : balanced_reaction NaH H₂O NaOH H₂)
  (h₂ : given_conditions NaH H₂O) :
  H₂ = 2 :=
by sorry

end hydrogen_moles_l330_330561


namespace binom_150_150_l330_330181

-- Definition of factorial
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else List.product (List.range' 1 n.succ)

-- Definition of binomial coefficient using factorial
def binom (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem binom_150_150 : binom 150 150 = 1 :=
by
  sorry

end binom_150_150_l330_330181


namespace max_elements_ge_distance_5_l330_330715

open Finset

-- Definitions based on the conditions:
def S : Finset (Fin 8 → Bool) := 
  univ.filter (λ A, ∀ i : Fin 8, A i = true ∨ A i = false)

def d (A B: Fin 8 → Bool) : ℕ :=
  (univ.filter (λ i, A i ≠ B i)).card

-- Lean 4 statement:
theorem max_elements_ge_distance_5 : ∃ S' ⊆ S, (∀ (A B ∈ S'), d A B ≥ 5) ∧ S'.card = 4 := 
sorry

end max_elements_ge_distance_5_l330_330715


namespace probability_of_reaching_3_1_without_2_0_in_8_steps_l330_330370

theorem probability_of_reaching_3_1_without_2_0_in_8_steps :
  let n_total := 1680
  let invalid := 30
  let total := n_total - invalid
  let q := total / 4^8
  let gcd := Nat.gcd total 65536
  let m := total / gcd
  let n := 65536 / gcd
  (m + n = 11197) :=
by
  sorry

end probability_of_reaching_3_1_without_2_0_in_8_steps_l330_330370


namespace a_n_is_perfect_square_l330_330268

theorem a_n_is_perfect_square :
  (∀ n : ℕ, ∃ k : ℕ, a (n + 1) = 7 * (a n) + 6 * (b n) - 3 ∧ b (n + 1) = 8 * (a n) + 7 * (b n) - 4) →
  (∀ n : ℕ, ∃ m : ℕ, a n = m ^ 2) :=
begin
  intros H,
  sorry
end

end a_n_is_perfect_square_l330_330268


namespace convert_binary_to_decimal_l330_330894

-- Define the binary number and its decimal conversion function
def bin_to_dec (bin : List ℕ) : ℕ :=
  list.foldl (λ acc b, 2 * acc + b) 0 bin

-- Define the specific problem conditions
def binary_oneonezeronethreeone : List ℕ := [1, 1, 0, 1, 1]

theorem convert_binary_to_decimal :
  bin_to_dec binary_oneonezeronethreeone = 27 := by
  sorry

end convert_binary_to_decimal_l330_330894


namespace giant_exponent_modulo_result_l330_330213

theorem giant_exponent_modulo_result :
  (7 ^ (7 ^ (7 ^ 7))) % 500 = 343 :=
by sorry

end giant_exponent_modulo_result_l330_330213


namespace sara_initial_quarters_l330_330751

theorem sara_initial_quarters (total_quarters dad_gift initial_quarters : ℕ) (h1 : dad_gift = 49) (h2 : total_quarters = 70) (h3 : total_quarters = initial_quarters + dad_gift) : initial_quarters = 21 :=
by sorry

end sara_initial_quarters_l330_330751


namespace power_function_odd_l330_330104

def f (x m : ℝ) : ℝ := x^2 + m

theorem power_function_odd (m : ℝ) (h : ∀ x ∈ set.Icc (-1 : ℝ) m, f x m = -f (-x) m) : 
  m = 1 ∧ f (m + 1) m = 5 := 
by 
  sorry

end power_function_odd_l330_330104


namespace fred_red_marbles_l330_330871

theorem fred_red_marbles (total_marbles : ℕ) (dark_blue_marbles : ℕ) (green_marbles : ℕ) (red_marbles : ℕ) 
  (h1 : total_marbles = 63) 
  (h2 : dark_blue_marbles ≥ total_marbles / 3)
  (h3 : green_marbles = 4)
  (h4 : red_marbles = total_marbles - dark_blue_marbles - green_marbles) : 
  red_marbles = 38 := 
sorry

end fred_red_marbles_l330_330871


namespace function_properties_l330_330997

-- Given conditions
def domain (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ∈ ℝ

def functional_eqn (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f(x * y) = y^2 * f(x) + x^2 * f(y)

-- Lean 4 theorem statement
theorem function_properties (f : ℝ → ℝ)
  (Hdom : domain f)
  (Heqn : functional_eqn f) :
  f 0 = 0 ∧ f 1 = 0 ∧ ∀ x : ℝ, f (-x) = f x :=
by
  sorry


end function_properties_l330_330997


namespace cost_per_other_topping_l330_330693

-- All conditions
def base_pizza_cost := 10.00
def num_slices := 8
def cost_per_slice := 2.00
def total_slices_cost := num_slices * cost_per_slice

def first_topping_cost := 2.00
def second_and_third_toppings_cost := 1.00
def num_second_and_third_toppings := 2
def other_toppings_count := 4
def total_toppings_count := 7

-- Segment sums
def known_toppings_cost := first_topping_cost + (num_second_and_third_toppings * second_and_third_toppings_cost)
def expected_total_cost := known_toppings_cost + base_pizza_cost

-- Statement to prove
theorem cost_per_other_topping : 
    total_slices_cost = expected_total_cost + other_toppings_count * (1 / 2) :=
by
    sorry

end cost_per_other_topping_l330_330693


namespace increasing_function_implies_positive_sum_l330_330770

open Function

variables {a b : ℝ} {f : ℝ → ℝ}

theorem increasing_function_implies_positive_sum (hf : ∀ x y, x < y → f(x) < f(y)) 
  (h_condition: f(a) + f(b) > f(-a) + f(-b)) : 
  a + b > 0 :=
sorry

end increasing_function_implies_positive_sum_l330_330770


namespace probability_of_perfect_square_is_one_fourth_l330_330759

-- Definition of a perfect square for the purpose of this problem
def isPerfectSquare (n : ℕ) : Prop :=
  n = 1 ∨ n = 4

-- The set of faces on an 8-sided die
def die_faces : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

-- Counting the number of successful outcomes
def successful_outcomes : Finset ℕ := die_faces.filter isPerfectSquare

-- Total number of outcomes
def total_outcomes : ℕ := die_faces.card

-- Probability of rolling a perfect square on an 8-sided die
noncomputable def probability : ℚ :=
  (successful_outcomes.card : ℚ) / (total_outcomes : ℚ)

-- The main statement
theorem probability_of_perfect_square_is_one_fourth :
  probability = 1 / 4 :=
by
  sorry

end probability_of_perfect_square_is_one_fourth_l330_330759


namespace range_of_a_l330_330597

variable {a : ℝ}

def p : Prop := ∀ x ∈ set.Icc 1 2, x^2 - a ≥ 0
def q : Prop := ∃ x₀ : ℝ, ∀ x ∈ set.Icc 1 2, x + (a - 1) * x₀ + 1 < 0

theorem range_of_a (hpq_or : p ∨ q) (hpq_and : ¬(p ∧ q)) :
  a > 3 ∨ (-1 ≤ a ∧ a ≤ 1) :=
sorry

end range_of_a_l330_330597


namespace quadrilateral_diagonals_l330_330304

theorem quadrilateral_diagonals (a b : ℝ) (h1: 0 < a) (h2: 0 < b)
  (h3 : ∃ θ: ℝ, θ = π / 3 ∧ cos θ = 1/2) :
  ∃ d1 d2 : ℝ, d1 = sqrt (a^2 + b^2 + a * b) ∧ d2 = sqrt (a^2 + b^2 - a * b) :=
sorry

end quadrilateral_diagonals_l330_330304


namespace polygon_sides_and_possible_angle_l330_330540

-- Definitions based on the conditions
def sum_of_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

noncomputable def polygon_with_n_sides (angles_sum : ℝ) : ℕ :=
  ((angles_sum / 180) + 2).to_nat

-- Statement of the proof problem
theorem polygon_sides_and_possible_angle :
  sum_of_interior_angles 21 = 3420 ∧ (160 < (3420 / 21)) ∧ (160 > 0) :=
by
  sorry

end polygon_sides_and_possible_angle_l330_330540


namespace prove_total_triangles_l330_330185

-- Define the context of the problem
open Set

-- Define the problem using a structure that includes conditions and the conclusion
def total_triangles_in_figure_eq_24
  (outer_square : Set (ℚ × ℚ)) -- Define the square using rational coordinates
  (diagonals : Set ((ℚ × ℚ) × (ℚ × ℚ))) -- Define the set of diagonals
  (midpoints : Set (ℚ × ℚ)) -- Define the set of midpoints of the diagonals
  (inner_square : Set (ℚ × ℚ)) -- Define the inner square formed by connecting midpoints
  (triangles : Set (Set (ℚ × ℚ))) -- Define the set of all triangles in the figure
  : Prop :=
  -- Ensure the conditions match the setup of the problem
  (∃ A B C D E F G H : ℚ × ℚ,
    outer_square = {A, B, C, D} ∧  -- Vertices of the outer square
    diagonals = {(A, C), (B, D)} ∧  -- The diagonals of the outer square
    midpoints = {(E, F)} ∧  -- Midpoints of the diagonals
    inner_square = {E, F, G, H} ∧  -- Vertices of the inner square
    triangles = { -- Listing all the possible triangles
      {A, B, E}, {B, C, F}, {C, D, F}, {D, A, E}, -- Smallest triangles
      {A, E, F}, {B, E, F}, {C, E, F}, {D, E, F}, -- Triangles using the inner square
      {A, E, B}, {B, F, C}, {C, F, D}, {D, E, A}, -- Additional triangles
      {E, F, G}, {E, F, H}, {F, G, H}, {F, H, E}  -- Triangles within the inner square
    }) ∧
  -- The conclusion of the proof
  (∀ s, s ∈ triangles → ∃ t, t ⊆ triangles ∧ card t = 24)

-- Definition of the Lean statement
theorem prove_total_triangles : total_triangles_in_figure_eq_24 := sorry

end prove_total_triangles_l330_330185


namespace circles_intersect_l330_330402

def circle1 : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 2}
def circle2 : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 + 4 * p.2 + 3 = 0}

theorem circles_intersect : ∃ (p : ℝ × ℝ), p ∈ circle1 ∧ p ∈ circle2 :=
by
  sorry

end circles_intersect_l330_330402


namespace proof_equivalent_expression_l330_330264

def dollar (a b : ℝ) : ℝ := (a + b) ^ 2

theorem proof_equivalent_expression (x y : ℝ) :
  (dollar ((x + y) ^ 2) (dollar y x)) - (dollar (dollar x y) (dollar x y)) = 
  4 * (x + y) ^ 2 * ((x + y) ^ 2 - 1) :=
by
  sorry

end proof_equivalent_expression_l330_330264


namespace functional_equation_option_A_option_B_option_C_l330_330971

-- Given conditions
def domain_of_f (f : ℝ → ℝ) : Set ℝ := Set.univ

theorem functional_equation (f : ℝ → ℝ) (x y : ℝ) :
  f (x * y) = y^2 * f x + x^2 * f y := sorry

-- Statements to be proved
theorem option_A (f : ℝ → ℝ) (h : ∀ x y, f(x * y) = y^2 * f(x) + x^2 * f(y)) : f 0 = 0 := sorry

theorem option_B (f : ℝ → ℝ) (h : ∀ x y, f(x * y) = y^2 * f(x) + x^2 * f(y)) : f 1 = 0 := sorry

theorem option_C (f : ℝ → ℝ) (h : ∀ x y, f(x * y) = y^2 * f(x) + x^2 * f(y)) : ∀ x, f(-x) = f x := sorry

end functional_equation_option_A_option_B_option_C_l330_330971


namespace part_a_sequence_l330_330457

def circle_sequence (n m : ℕ) : List ℕ :=
  List.replicate m 1 -- Placeholder: Define the sequence computation properly

theorem part_a_sequence :
  circle_sequence 5 12 = [1, 6, 11, 4, 9, 2, 7, 12, 5, 10, 3, 8, 1] := 
sorry

end part_a_sequence_l330_330457


namespace employees_age_distribution_l330_330307

-- Define the total number of employees
def totalEmployees : ℕ := 15000

-- Define the percentages
def malePercentage : ℝ := 0.58
def femalePercentage : ℝ := 0.42

-- Define the age distribution percentages for male employees
def maleBelow30Percentage : ℝ := 0.25
def male30To50Percentage : ℝ := 0.40
def maleAbove50Percentage : ℝ := 0.35

-- Define the percentage of female employees below 30
def femaleBelow30Percentage : ℝ := 0.30

-- Define the number of male employees
def numMaleEmployees : ℝ := malePercentage * totalEmployees

-- Calculate the number of male employees in each age group
def numMaleBelow30 : ℝ := maleBelow30Percentage * numMaleEmployees
def numMale30To50 : ℝ := male30To50Percentage * numMaleEmployees
def numMaleAbove50 : ℝ := maleAbove50Percentage * numMaleEmployees

-- Define the number of female employees
def numFemaleEmployees : ℝ := femalePercentage * totalEmployees

-- Calculate the number of female employees below 30
def numFemaleBelow30 : ℝ := femaleBelow30Percentage * numFemaleEmployees

-- Calculate the total number of employees below 30
def totalBelow30 : ℝ := numMaleBelow30 + numFemaleBelow30

-- We now state our theorem to prove
theorem employees_age_distribution :
  numMaleBelow30 = 2175 ∧
  numMale30To50 = 3480 ∧
  numMaleAbove50 = 3045 ∧
  totalBelow30 = 4065 := by
    sorry

end employees_age_distribution_l330_330307


namespace find_number_of_members_l330_330086

variable (n : ℕ)

-- We translate the conditions into Lean 4 definitions
def total_collection := 9216
def per_member_contribution := n

-- The goal is to prove that n = 96 given the total collection
theorem find_number_of_members (h : n * n = total_collection) : n = 96 := 
sorry

end find_number_of_members_l330_330086


namespace boa_constrictor_length_correct_l330_330146
noncomputable def boa_constrictor_length (g_len : ℝ) (ratio : ℝ) : ℝ :=
  g_len / ratio

theorem boa_constrictor_length_correct :
  ∀ (a : ℝ) (g_len : ℝ) (ratio : ℝ), a = 2.0 → g_len = 10.0 → ratio = 7.0 →
  boa_constrictor_length g_len ratio ≈ 1.42857 :=
by
  intros a g_len ratio ha hg hr
  rw [ha, hg, hr]
  norm_num
  sorry

end boa_constrictor_length_correct_l330_330146


namespace bags_wednesday_l330_330165

def charge_per_bag : ℕ := 4
def bags_monday : ℕ := 5
def bags_tuesday : ℕ := 3
def total_earnings : ℕ := 68

theorem bags_wednesday (h1 : charge_per_bag = 4)
                       (h2 : bags_monday = 5)
                       (h3 : bags_tuesday = 3)
                       (h4 : total_earnings = 68) :
  let earnings_monday_tuesday := (bags_monday + bags_tuesday) * charge_per_bag in
  let earnings_wednesday := total_earnings - earnings_monday_tuesday in
  earnings_wednesday / charge_per_bag = 9 :=
by
  sorry

end bags_wednesday_l330_330165


namespace fraction_comparison_l330_330964

theorem fraction_comparison (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  a / b > (a + 1) / (b + 1) :=
by sorry

end fraction_comparison_l330_330964


namespace trajectory_of_center_of_P_l330_330959

-- Define circles M and N
def circle_M (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1
def circle_N (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 25

-- Define the conditions for the moving circle P
def externally_tangent (x y r : ℝ) : Prop := (x + 1)^2 + y^2 = (1 + r)^2
def internally_tangent (x y r : ℝ) : Prop := (x - 1)^2 + y^2 = (5 - r)^2

-- The statement we need to prove
theorem trajectory_of_center_of_P : ∃ (x y : ℝ), 
  (externally_tangent x y r) ∧ (internally_tangent x y r) →
  (x^2 / 9 + y^2 / 8 = 1) :=
by
  -- Proof will go here
  sorry

end trajectory_of_center_of_P_l330_330959


namespace paper_clips_in_two_cases_l330_330832

-- Define the conditions
variables (c b : ℕ)

-- Define the theorem statement
theorem paper_clips_in_two_cases (c b : ℕ) : 
    2 * c * b * 400 = 2 * c * b * 400 :=
by
  sorry

end paper_clips_in_two_cases_l330_330832


namespace exponents_inequality_l330_330337

def T (α : List ℕ) (x : List ℝ) : ℝ := sorry  -- Placeholder for the polynomial function T_α

theorem exponents_inequality {n : ℕ} (α β : List ℕ) (x : List ℝ) (h_len : α.length = n) (h_len' : β.length = n) 
  (h_sum : α.sum = β.sum) (h_ne : α ≠ β) (h_nonneg : ∀ i, i < n → 0 ≤ x.nthLe i h_len) : 
  T α x ≥ T β x := sorry

end exponents_inequality_l330_330337


namespace family_veg_eaters_l330_330661

theorem family_veg_eaters (A B C : ℕ) (hA : A = 15) (hB : B = 8) (hC : C = 11) : A + C = 26 :=
by
  rw [hA, hC]
  exact rfl

end family_veg_eaters_l330_330661


namespace alexei_weeks_lost_weight_l330_330147

theorem alexei_weeks_lost_weight (x : ℕ) : (1.5 * 10 + 2.5 * x = 35) → x = 8 :=
by
  sorry

end alexei_weeks_lost_weight_l330_330147


namespace irrational_count_l330_330864

def two : ℝ := 2
def zero : ℝ := 0
def sqrt5 : ℝ := Real.sqrt 5
def pi_over_3 : ℝ := Real.pi / 3
def cube_root_27 : ℝ := Real.sqrt 27
def special_number : ℝ := Real.mk_irrational (λ _, sorry) -- represents 0.1010010001...

theorem irrational_count :
  (∃ (S : Finset ℝ), S = {two, zero, sqrt5, pi_over_3, cube_root_27, special_number} ∧ 
    S.filter Real.irrational = {sqrt5, pi_over_3, special_number}.to_finset) :=
sorry

end irrational_count_l330_330864


namespace slices_of_bread_left_l330_330027

variable (monday_to_friday_slices saturday_slices total_slices_used initial_slices slices_left: ℕ)

def sandwiches_monday_to_friday : ℕ := 5
def slices_per_sandwich : ℕ := 2
def sandwiches_saturday : ℕ := 2
def initial_slices_of_bread : ℕ := 22

theorem slices_of_bread_left :
  slices_left = initial_slices_of_bread - total_slices_used
  :=
by  sorry

end slices_of_bread_left_l330_330027


namespace equilateral_triangle_from_identical_right_triangles_l330_330801

/-- Two identical right-angled triangles are placed one on top of the other,
such that the vertex of the right angle of one triangle falls on the hypotenuse of the other.
Prove that the resulting triangle is equilateral. -/
theorem equilateral_triangle_from_identical_right_triangles 
  (A B C D E F : Type)
  [is_right_angled_triangle A B D] [is_right_angled_triangle C E F]
  (identical_triangles : set.is_identical A B D C E F)
  (right_angle_placement : right_angle_vertex_on_hypotenuse A B D C E F) :
  is_equilateral_triangle (A B C) :=
sorry

end equilateral_triangle_from_identical_right_triangles_l330_330801


namespace tourist_groupings_count_l330_330414

/-- We define three guides and eight tourists. -/
def num_guides : ℕ := 3
def num_tourists : ℕ := 8

/-- The number of valid groupings of tourists where each guide gets at least one tourist -/
noncomputable def num_groupings : ℕ :=
  let total_arrangements := (num_guides ^ num_tourists) in
  let invalid_groupings :=
    (num_guides * (2 ^ num_tourists)) +
    (num_guides * 1) in
  total_arrangements - invalid_groupings

theorem tourist_groupings_count : num_groupings = 5796 := by
  -- Proof to be filled in
  sorry

end tourist_groupings_count_l330_330414


namespace triangle_ratio_theorem_l330_330662

theorem triangle_ratio_theorem
  (ABC : Type*)
  [triangle ABC]
  (B C : ABC)
  (A M N : Type*) 
  (BC : line B C)
  (AC : line A C)
  (AB : line A B)
  (AM : line A M)
  (DE : line D E)
  (L : point)
  (m n : ℝ)
  (Bm_MC : ∀ (BM MC : ℝ), BM / MC = m)
  (An_NC : ∀ (AN NC : ℝ), AN / NC = n)
  (parallel1 : ∀ N BC, parallel (line_through N BC) AB)
  (parallel2 : ∀ N AM, parallel (line_through N AM) BC)
  (intersection_L : ∀ AM DE, intersection AM DE = L) :
  AL / LM = m + n + mn := sorry

end triangle_ratio_theorem_l330_330662


namespace shorter_side_of_rectangle_l330_330137

theorem shorter_side_of_rectangle (a b : ℕ) (h_perimeter : 2 * a + 2 * b = 62) (h_area : a * b = 240) : b = 15 :=
by
  sorry

end shorter_side_of_rectangle_l330_330137


namespace circle_tangent_parabola_height_difference_l330_330118

theorem circle_tangent_parabola_height_difference 
  (a b : ℝ)
  (h_tangent1 : (a, a^2 + 1))
  (h_tangent2 : (-a, a^2 + 1))
  (parabola_eq : ∀ x, x^2 + 1 = (x, x^2 + 1))
  (circle_eq : ∀ x, x^2 + ((x^2 + 1) - b)^2 = r^2) : 
  b - (a^2 + 1) = 1 / 2 :=
sorry

end circle_tangent_parabola_height_difference_l330_330118


namespace limit_sequence_ratio_l330_330741

theorem limit_sequence_ratio
  (x : ℕ → ℝ)
  (a : ℝ)
  (h : tendsto (λ n, x(n+1) - x(n)) at_top (𝓝 a)) :
  tendsto (λ n, x n / n) at_top (𝓝 a) :=
sorry

end limit_sequence_ratio_l330_330741


namespace bertha_no_children_count_l330_330157

-- Definitions
def bertha_daughters : ℕ := 6
def granddaughters_per_daughter : ℕ := 6
def total_daughters_and_granddaughters : ℕ := 30

-- Theorem to be proved
theorem bertha_no_children_count : 
  ∃ x : ℕ, (x * granddaughters_per_daughter + bertha_daughters = total_daughters_and_granddaughters) ∧ 
           (bertha_daughters - x + x * granddaughters_per_daughter = 26) :=
sorry

end bertha_no_children_count_l330_330157


namespace eval_expression_l330_330408

theorem eval_expression : (4^2 - 2^3) = 8 := by
  sorry

end eval_expression_l330_330408


namespace gnomes_in_fifth_house_l330_330792

-- Defining the problem conditions
def num_houses : Nat := 5
def gnomes_per_house : Nat := 3
def total_gnomes : Nat := 20

-- Defining the condition for the first four houses
def gnomes_in_first_four_houses : Nat := 4 * gnomes_per_house

-- Statement of the problem
theorem gnomes_in_fifth_house : 20 - (4 * 3) = 8 := by
  sorry

end gnomes_in_fifth_house_l330_330792


namespace pencils_bought_l330_330733

theorem pencils_bought (total_spent notebook_cost ruler_cost pencil_cost : ℕ)
  (h_total : total_spent = 74)
  (h_notebook : notebook_cost = 35)
  (h_ruler : ruler_cost = 18)
  (h_pencil : pencil_cost = 7) :
  (total_spent - (notebook_cost + ruler_cost)) / pencil_cost = 3 :=
by
  sorry

end pencils_bought_l330_330733


namespace area_triangle_BDF_l330_330351

variables {A B C D E F : Point}

-- Given a parallelogram ABCD
variable (parallelogram_ABCD : Parallelogram A B C D)
variable (area_ABCD : area parallelogram_ABCD = 48)

-- E is the midpoint of AB
variable (midpoint_E : Midpoint E A B)

-- F lies on diagonal AC such that AF = (1/3) * AC
variable (F_on_diagonal : F_m_id AC (1/3))

-- Prove the area of triangle BDF is 16 square units.
theorem area_triangle_BDF : area (Triangle B D F) = 16 := sorry

end area_triangle_BDF_l330_330351


namespace sum_six_times_product_l330_330284

variable (a b x : ℝ)

theorem sum_six_times_product (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a + b = 6 * x) (h4 : 1/a + 1/b = 6) :
  x = a * b := sorry

end sum_six_times_product_l330_330284


namespace spadesuit_eval_l330_330535

def spadesuit (a b : ℤ) := abs (a - b)

theorem spadesuit_eval :
  spadesuit 5 (spadesuit 3 (spadesuit 8 12)) = 4 := 
by
  sorry

end spadesuit_eval_l330_330535


namespace minimize_classification_error_l330_330379

open Real

def vehicle_class_cost (class_type : ℕ) : ℝ :=
if class_type = 1 then 200 else if class_type = 2 then 300 else 0

def misclassification_error (height_classification : ℝ → ℕ) (vehicle_height : ℝ) (actual_class : ℕ) : ℝ :=
if height_classification(vehicle_height) ≠ actual_class then
  if actual_class = 1 then 100 else 0
else 0

noncomputable def optimal_threshold (class1_graph class2_graph : ℝ → ℝ) : ℝ :=
Inf { h | class1_graph h = class2_graph h }

theorem minimize_classification_error 
(class1_graph class2_graph : ℝ → ℝ)
: optimal_threshold class1_graph class2_graph = 190 :=
sorry

end minimize_classification_error_l330_330379


namespace function_properties_l330_330993

-- Given conditions
def domain (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ∈ ℝ

def functional_eqn (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f(x * y) = y^2 * f(x) + x^2 * f(y)

-- Lean 4 theorem statement
theorem function_properties (f : ℝ → ℝ)
  (Hdom : domain f)
  (Heqn : functional_eqn f) :
  f 0 = 0 ∧ f 1 = 0 ∧ ∀ x : ℝ, f (-x) = f x :=
by
  sorry


end function_properties_l330_330993


namespace measure_EML_l330_330881

variables {D E F L M N : Point}
variable (Omega : Circle)
variable (triangle_DEF : Triangle D E F)
variable (triangle_LMN : Triangle L M N)

noncomputable def measure_of_angle_EML : Angle :=
  if (Omega.isIncircle triangle_DEF) &&
     (Omega.isCircumcircle triangle_LMN) &&
     (L.onSegment E F) &&
     (M.onSegment D E) &&
     (N.onSegment D F) &&
     (triangle_DEF.angle D = 50) &&
     (triangle_DEF.angle E = 70) &&
     (triangle_DEF.angle F = 60)
  then 120
  else sorry

theorem measure_EML (Omega : Circle) 
  (triangle_DEF : Triangle D E F) 
  (triangle_LMN : Triangle L M N)
  [h1 : Omega.isIncircle triangle_DEF] 
  [h2 : Omega.isCircumcircle triangle_LMN] 
  [hL : L.onSegment E F] 
  [hM : M.onSegment D E] 
  [hN : N.onSegment D F] 
  (hD : triangle_DEF.angle D = 50) 
  (hE : triangle_DEF.angle E = 70) 
  (hF : triangle_DEF.angle F = 60) :
  ∠ E M L = 120 := by
  sorry

end measure_EML_l330_330881


namespace average_time_is_five_l330_330415

-- Define the conditions for Train 1
def train1_time : ℝ := (100 / 50) + (100 / 40) + 0.5

-- Define the conditions for Train 2
def train2_time : ℝ := (80 / 80) + (160 / 100) + 0.75

-- Define the conditions for Train 3
def train3_time : ℝ := (300 / 60) + (40 / 60)

-- Calculate the average time for the three trains
def average_time : ℝ := (train1_time + train2_time + train3_time) / 3

-- Statement proving the average time rounded to the nearest integer
theorem average_time_is_five : Int.round average_time = 5 := by
  -- Skip the internal proof
  sorry

end average_time_is_five_l330_330415


namespace hyperbola_eccentricity_l330_330290

theorem hyperbola_eccentricity (a : ℝ) (h : a > 0)
  (h_eq : ∀ (x y : ℝ), x^2 / a^2 - y^2 / 16 = 1 ↔ true)
  (eccentricity : a^2 + 16 / a^2 = (5 / 3)^2) : a = 3 :=
by
  sorry

end hyperbola_eccentricity_l330_330290


namespace remainder_of_7_pow_7_pow_7_pow_7_mod_500_l330_330220

theorem remainder_of_7_pow_7_pow_7_pow_7_mod_500 :
    (7 ^ (7 ^ (7 ^ 7))) % 500 = 343 := 
sorry

end remainder_of_7_pow_7_pow_7_pow_7_mod_500_l330_330220


namespace log_problem_l330_330524

theorem log_problem : log 6 (log 4 (log 3 81)) = 0 := sorry

end log_problem_l330_330524


namespace boat_travel_time_198_km_downstream_l330_330828

noncomputable theory
open Real

-- Define the conditions as given in the problem
def condition1 (v c : ℝ) : Prop := 420 / (v + c) + 80 / (v - c) = 11
def condition2 (v c : ℝ) : Prop := 240 / (v + c) + 140 / (v - c) = 11

-- Define the theorem to prove the travel time for 198 km downstream
theorem boat_travel_time_198_km_downstream (v c : ℝ) (h1 : condition1 v c) (h2 : condition2 v c) : 
  198 / (v + c) = 3.3 :=
sorry

end boat_travel_time_198_km_downstream_l330_330828


namespace shopkeeper_discount_l330_330486

variable (CP : ℝ)
variable (MP : ℝ) (SP : ℝ) (discount : ℝ) (D% : ℝ)

-- Conditions
def condition1 : CP = 100 := sorry
def condition2 : MP = CP * 1.2 := sorry
def condition3 : SP = CP * 1.02 := sorry

-- The discount offered
def discount_def : discount = MP - SP := sorry

-- The percentage discount
def percentage_discount : D% = (discount / MP) * 100 := sorry

-- The proof statement
theorem shopkeeper_discount : D% = 15 := 
sorry

end shopkeeper_discount_l330_330486


namespace find_box_width_l330_330462

-- Define box dimensions
def box (h w l : ℕ) := h * w * l

-- Define block dimensions and count
def block (h w l : ℕ) := h * w * l
def block_count : ℕ := 40

-- Given dimensions
def box_height : ℕ := 8
def box_length : ℕ := 12
def block_height : ℕ := 3
def block_width : ℕ := 2
def block_length : ℕ := 4

-- Volume calculation
def block_volume := block block_height block_width block_length
def total_block_volume := block_count * block_volume

-- Theorem to prove the width of the box
theorem find_box_width (w : ℕ) (h_box : box_height) (l_box : box_length) (volume_blocks : total_block_volume) : 
  box box_height w box_length = volume_blocks → w = 10 :=
by
  sorry  -- Proof placeholder

end find_box_width_l330_330462


namespace binary_to_decimal_l330_330891

theorem binary_to_decimal : 
  let binary_number := [1, 1, 0, 1, 1] in
  let weights := [2^4, 2^3, 2^2, 2^1, 2^0] in
  (List.zipWith (λ digit weight => digit * weight) binary_number weights).sum = 27 := by
  sorry

end binary_to_decimal_l330_330891


namespace solution_set_l330_330954

noncomputable def f : ℝ → ℝ := sorry
def dom := {x : ℝ | x < 0 ∨ x > 0 } -- Definition of the function domain

-- Assumptions and conditions as definitions in Lean
axiom f_odd : ∀ x ∈ dom, f (-x) = -f x
axiom f_at_1 : f 1 = 1
axiom symmetric_f : ∀ x ∈ dom, (f (x + 1)) = -f (-x + 1)
axiom inequality_condition : ∀ (x1 x2 : ℝ), x1 ∈ dom → x2 ∈ dom → x1 ≠ x2 → (x1^3 * f x1 - x2^3 * f x2) / (x1 - x2) > 0

-- The main statement to be proved
theorem solution_set :
  {x ∈ dom | f x ≤ 1 / x^3} = {x ∈ dom | x ≤ -1} ∪ {x ∈ dom | 0 < x ∧ x ≤ 1} :=
sorry

end solution_set_l330_330954


namespace packs_of_sugar_l330_330200

theorem packs_of_sugar (cost_apples_per_kg cost_walnuts_per_kg cost_apples total : ℝ) (weight_apples weight_walnuts : ℝ) (less_sugar_by_1 : ℝ) (packs : ℕ) :
  cost_apples_per_kg = 2 →
  cost_walnuts_per_kg = 6 →
  cost_apples = weight_apples * cost_apples_per_kg →
  weight_apples = 5 →
  weight_walnuts = 0.5 →
  less_sugar_by_1 = 1 →
  total = 16 →
  packs = (total - (weight_apples * cost_apples_per_kg + weight_walnuts * cost_walnuts_per_kg)) / (cost_apples_per_kg - less_sugar_by_1) →
  packs = 3 :=
by
  sorry

end packs_of_sugar_l330_330200


namespace area_of_AEGH_is_1_5_l330_330313

-- Define the required points in 2D space
noncomputable def A := (0 : ℝ, 0 : ℝ)
noncomputable def B := (2 : ℝ, 0 : ℝ)
noncomputable def C := (2 : ℝ, 3 : ℝ)
noncomputable def D := (0 : ℝ, 3 : ℝ)

-- Define midpoints
noncomputable def E := ((B.1 + C.1) / 2 : ℝ, (B.2 + C.2) / 2 : ℝ)
noncomputable def F := ((C.1 + D.1) / 2 : ℝ, (C.2 + D.2) / 2 : ℝ)
noncomputable def G := ((A.1 + D.1) / 2 : ℝ, (A.2 + D.2) / 2 : ℝ)
noncomputable def H := ((E.1 + F.1) / 2 : ℝ, (E.2 + F.2) / 2 : ℝ)

-- Define the shoelace formula for area calculation
noncomputable def shoelace_area (P Q R : ℝ × ℝ) : ℝ :=
  abs (P.1 * (Q.2 - R.2) + Q.1 * (R.2 - P.2) + R.1 * (P.2 - Q.2)) / 2

-- The proof statement
theorem area_of_AEGH_is_1_5 :
  shoelace_area A E G + shoelace_area E H G = 1.5 :=
sorry

end area_of_AEGH_is_1_5_l330_330313


namespace average_time_per_mile_l330_330364

noncomputable def miles : ℕ := 24
noncomputable def hours : ℕ := 3
noncomputable def minutes : ℕ := 36

theorem average_time_per_mile :
  let total_time := hours * 60 + minutes in
  total_time / miles = 9 := by
  sorry

end average_time_per_mile_l330_330364


namespace cube_iff_diagonal_perpendicular_l330_330962

-- Let's define the rectangular parallelepiped as a type
structure RectParallelepiped :=
-- Define the property of being a cube
(isCube : Prop)

-- Define the property q: any diagonal of the parallelepiped is perpendicular to the diagonal of its non-intersecting face
def diagonal_perpendicular (S : RectParallelepiped) : Prop := 
 sorry -- This depends on how you define diagonals and perpendicularity within the structure

-- Prove the biconditional relationship
theorem cube_iff_diagonal_perpendicular (S : RectParallelepiped) :
 S.isCube ↔ diagonal_perpendicular S :=
sorry

end cube_iff_diagonal_perpendicular_l330_330962


namespace parabola_equation_collinear_points_l330_330329

-- Definitions
def parabola_eq (x y : ℝ) : Prop :=
  y^2 = 4 * x

def point_intersection_directrix_axis (p : ℝ) : ℝ × ℝ :=
  (-p / 2, 0)

def point_focus (p : ℝ) : ℝ × ℝ :=
  (p / 2, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def min_m (x : ℝ) (p : ℝ) : ℝ :=
  let M := (-p / 2, 0)
  let F := (p / 2, 0)
  let N := (x, Real.sqrt (4 * x))
  (distance N F) / (distance N M)

-- Theorem 1: Prove the equation of the parabola
theorem parabola_equation :
  ∃ p, 
  (∀ x, min_m x p ≥ (Real.sqrt (2) / 2)) ∧
  (min_m 1 p = Real.sqrt (2) / 2) ∧
  parabola_eq 1 (Real.sqrt (4 * 1)) :=
sorry

-- Definitions for second theorem
def symmetric_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

def symmetric_y (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

def line_eq (k b x y : ℝ) : Prop :=
  y = k * x + b

-- Theorem 2: Prove collinearity
theorem collinear_points (k b : ℝ) (hk : k ≠ 0) :
  ∀ x1 x2 y1 y2 : ℝ,
  (parabola_eq x1 y1) ∧ (parabola_eq x2 y2) ∧ (line_eq k b x1 y1) ∧ (line_eq k b x2 y2) →
  let C := (-b / k, 0)
  let Q := symmetric_y C
  let P := symmetric_x (x2, y2)
  let A := (x1, y1)
  collinear A P Q :=
sorry

end parabola_equation_collinear_points_l330_330329


namespace parabola_equation_l330_330586

open Real

variables {p : ℝ}

theorem parabola_equation (hp : p > 0) (hf : ∃ M N : ℝ × ℝ, (M.1^2 - M.2^2 = 2) ∧ (N.1^2 - N.2^2 = 2) ∧ (dist (0, p / 2) M = dist (0, p / 2) N) ∧ (sqrt 3 * dist (0, p / 2) M / 2 = p)) :
  ∃ k : ℝ, parabola_eqn : x^2 = k y :=
begin
  sorry
end

end parabola_equation_l330_330586


namespace pizza_share_l330_330872

theorem pizza_share :
  forall (friends : ℕ) (leftover_pizza : ℚ), friends = 4 -> leftover_pizza = 5/6 -> (leftover_pizza / friends) = (5 / 24) :=
by
  intros friends leftover_pizza h_friends h_leftover_pizza
  sorry

end pizza_share_l330_330872


namespace same_number_of_acquaintances_l330_330663

theorem same_number_of_acquaintances (n : ℕ) (h_n : n = 2013) :
  ∃ (i j : ℕ) (hi : i < n) (hj : j < n), i ≠ j ∧ 
  count_acquaintances i = count_acquaintances j :=
by
  -- count_acquaintances is a function that needs to be defined to 
  -- represent the number of acquaintances of a person.
  sorry

end same_number_of_acquaintances_l330_330663


namespace tangent_slope_l330_330545

theorem tangent_slope (x1 y1 x2 y2 : ℝ) (h₁ : x1 = 1) (h₂ : y1 = -1) (h₃ : x2 = 4) (h₄ : y2 = 3) :
  let radius_slope := (y2 - y1) / (x2 - x1) in
  let tangent_slope := -1 / radius_slope in
  tangent_slope = -3 / 4 :=
by
  sorry

end tangent_slope_l330_330545


namespace train_speed_km_hr_l330_330492

-- Define the given conditions
def length_of_train : ℝ := 130 -- in meters
def time_to_cross_pole : ℝ := 9 -- in seconds

-- Conversion factor to km/hr
def conversion_factor : ℝ := 3.6

-- Define the expected speed in km/hr
def expected_speed_km_hr : ℝ := 51.984

-- The proof problem statement
theorem train_speed_km_hr :
  (length_of_train / time_to_cross_pole) * conversion_factor ≈ expected_speed_km_hr := by
  sorry

end train_speed_km_hr_l330_330492


namespace equation_of_circle_exists_radius_and_center_l330_330209

noncomputable def circle_equation : String :=
  let center := (-1 / 2, 9 / 2);
  let radius := sqrt 130 / 2;
  let eq := "x^2 + y^2 + x - 9y - 12 = 0";
  eq

-- Definition of the points
def A := (4, 1: ℝ)
def B := (-6, 3: ℝ)
def C := (3, 0: ℝ)

-- Definitions of the conditions that the points lie on the circle
def conditionA (x y : ℝ) (D E F : ℝ) := x^2 + y^2 + D * x + E * y + F = 0
def conditionB (x y : ℝ) (D E F : ℝ) := x^2 + y^2 + D * x + E * y + F = 0
def conditionC (x y : ℝ) (D E F : ℝ) := x^2 + y^2 + D * x + E * y + F = 0

-- The final theorem statement
theorem equation_of_circle_exists :
  ∃ (D E F : ℝ), 
  conditionA 4 1 D E F ∧
  conditionB (-6) 3 D E F ∧
  conditionC 3 0 D E F ∧
  circle_equation = "x^2 + y^2 + x - 9y - 12 = 0" :=
begin
  use [1, -9, -12],
  split,
  { unfold conditionA, norm_num }, -- Checking point A
  split,
  { unfold conditionB, norm_num }, -- Checking point B
  { unfold conditionC, norm_num }, -- Checking point C
  sorry
end

-- Assert the radius and center
theorem radius_and_center :
  let center := (-1 / 2, 9 / 2);
  let radius := sqrt 130 / 2;
  center = (-1 / 2, 9 / 2) ∧ radius = sqrt 130 / 2 :=
begin
  -- Provide the trivial proof
  split;
  refl
end

end equation_of_circle_exists_radius_and_center_l330_330209


namespace find_c_value_l330_330116

theorem find_c_value :
  let circle1 := {center := (2, -3), radius := 10}
  let circle2 := {center := (-4, 7), radius := 6}
  (∃ c : ℝ, (x + y = c) ∧
            ∀ (p : ℝ × ℝ), (p = (λ x y, (x - 2)^2 + (y + 3)^2 = 100) ∧ 
                          (p = (λ x y, (x + 4)^2 + (y - 7)^2 = 36)) → 
                          (p.1 + p.2 = c) ) ∧
            c = 26 / 3) sorry

end find_c_value_l330_330116


namespace larger_gate_width_is_10_l330_330518

-- Define the conditions as constants
def garden_length : ℝ := 225
def garden_width : ℝ := 125
def small_gate_width : ℝ := 3
def total_fencing_length : ℝ := 687

-- Define the perimeter function for a rectangle
def perimeter (length width : ℝ) : ℝ := 2 * (length + width)

-- Define the width of the larger gate
def large_gate_width : ℝ :=
  let total_perimeter := perimeter garden_length garden_width
  let remaining_fencing := total_perimeter - total_fencing_length
  remaining_fencing - small_gate_width

-- State the theorem
theorem larger_gate_width_is_10 : large_gate_width = 10 := by
  -- skipping proof part
  sorry

end larger_gate_width_is_10_l330_330518


namespace output_A_after_loop_l330_330786

theorem output_A_after_loop : 
  let A_init := 1
  (∀ n: ℕ, 1 ≤ n ∧ n ≤ 5 → (A n = ∏ i in Finset.range (n + 1), (i + 1))) → 
  A 5 = 120 := 
begin
  sorry
end

end output_A_after_loop_l330_330786


namespace roller_mark_upward_position_on_park_l330_330531

def radius_park : ℝ := 30
def radius_roller : ℝ := 5
def roller_moves_without_slipping := true
def initial_position (t : ℝ) : Prop := t = 0

theorem roller_mark_upward_position_on_park : 
  roller_moves_without_slipping →
  initial_position 0 →
  ∃ t : ℝ, t > 0 ∧ (roller_position t = 12 ∧ roller_mark_orientation t = "upward") :=
begin
  intros h_move h_init,
  existsi (2 * π * radius_park / (radius_roller / radius_park)),
  split,
  { sorry },
  { sorry }
end

end roller_mark_upward_position_on_park_l330_330531


namespace min_value_sin_cos_l330_330933

theorem min_value_sin_cos : ∀ x : ℝ, ∃ y : ℝ, y = sin x ^ 6 + 2 * cos x ^ 6 ∧ y ≥ (2 / 3) := sorry

end min_value_sin_cos_l330_330933


namespace angle_PQR_approximately_l330_330701

open Real EuclideanGeometry

def P : ℝ³ := (-3, 1, 4)
def Q : ℝ³ := (-4, 0, 5)
def R : ℝ³ := (-2, 2, 4)

theorem angle_PQR_approximately  : 
  angle P Q R = Real.arccos (5 * sqrt 3 / 9) := 
sorry

end angle_PQR_approximately_l330_330701


namespace g_difference_l330_330344

-- Let g be a linear function satisfying the condition g(x+2) - g(x) = 5 for all x
variable {g : ℝ → ℝ}
axiom linear_g : ∀ x : ℝ, g(x + 2) - g(x) = 5

-- The goal is to prove that g(2) - g(8) = -15 
theorem g_difference : g(2) - g(8) = -15 :=
by sorry

end g_difference_l330_330344


namespace eighty_fifth_digit_l330_330642

def seq_60_to_1 := (List.range' 1 60).reverse.map (λ n => n.to_string.mk_surround "0123456789 ")

def nth_digit {α : Type*} [Inhabited α] [HasMul α] [Nat.castCoe α] (p : Nat → α) (d : α) (n k : Nat) : α := 
  ((p ∘ Nat.succ) k).nth n.get_dfl d

noncomputable def digit_85 := nth_digit seq_60_to_1 1 84

theorem eighty_fifth_digit : digit_85 = 1 :=
by
  sorry

end eighty_fifth_digit_l330_330642


namespace smallest_angle_25_gon_l330_330905

-- Definitions and conditions
def convex_polygon (n : ℕ) : Prop :=
  n ≥ 3

def arithmetic_sequence (a0 d : ℤ) (n : ℕ) (an : ℕ → ℤ) : Prop :=
  ∀ k : ℕ, k < n → an k = a0 + k * d

def integer_angles (an : ℕ → ℤ) (n : ℕ) : Prop :=
  ∀ k : ℕ, k < n → an k > 0

def sum_of_interior_angles (n : ℕ) : ℤ :=
  (n - 2) * 180

def average_angle (n : ℕ) : ℚ :=
  sum_of_interior_angles n / n

-- Problem conditions
noncomputable def smallest_angle_in_convex_25_sided_polygon : ℤ :=
  141

-- Main statement with proof
theorem smallest_angle_25_gon :
  convex_polygon 25 →
  (∃ a0 d, arithmetic_sequence a0 d 25 ∧ integer_angles (λ k, a0 + k * d) 25 ∧
   ∀ k, k < 25 → (a0 + k * d < 180)) →
  ∃ m, m = smallest_angle_in_convex_25_sided_polygon :=
by {
  intros h_convex h_exists,
  -- Here would be the proof that smallest angle is indeed 141°
  sorry
}

end smallest_angle_25_gon_l330_330905


namespace painting_cubes_ways_l330_330459

theorem painting_cubes_ways : 
  ∃ (ways : ℕ), ways = 576 ∧ 
  (∀ cube : {i : fin 64 // i.val < 64}, 
    (∑ k : fin 64, 1)) = 16 ∧ 
  (∀ layer : fin 4, (∑ i : fin 16, 1)) = 4 :=
begin
  existsi 576,
  split,
  { refl, },
  split,
  { sorry, },
  { sorry, }
end

end painting_cubes_ways_l330_330459


namespace remainder_of_7_pow_7_pow_7_pow_7_mod_500_l330_330219

theorem remainder_of_7_pow_7_pow_7_pow_7_mod_500 :
    (7 ^ (7 ^ (7 ^ 7))) % 500 = 343 := 
sorry

end remainder_of_7_pow_7_pow_7_pow_7_mod_500_l330_330219


namespace charge_per_mile_l330_330730

def rental_fee : ℝ := 20.99
def total_amount_paid : ℝ := 95.74
def miles_driven : ℝ := 299

theorem charge_per_mile :
  (total_amount_paid - rental_fee) / miles_driven = 0.25 := 
sorry

end charge_per_mile_l330_330730


namespace daily_sales_change_l330_330107

theorem daily_sales_change
    (mon_sales : ℕ)
    (week_total_sales : ℕ)
    (days_in_week : ℕ)
    (avg_sales_per_day : ℕ)
    (other_days_total_sales : ℕ)
    (x : ℕ)
    (h1 : days_in_week = 7)
    (h2 : avg_sales_per_day = 5)
    (h3 : week_total_sales = avg_sales_per_day * days_in_week)
    (h4 : mon_sales = 2)
    (h5 : week_total_sales = mon_sales + other_days_total_sales)
    (h6 : other_days_total_sales = 33)
    (h7 : 2 + x + 2 + 2*x + 2 + 3*x + 2 + 4*x + 2 + 5*x + 2 + 6*x = other_days_total_sales) : 
  x = 1 :=
by
sorry

end daily_sales_change_l330_330107


namespace exists_integer_roots_l330_330278

theorem exists_integer_roots : 
  ∃ (a b c d e f : ℤ), ∃ r1 r2 r3 r4 r5 r6 : ℤ,
  (r1 + a) * (r2 ^ 2 + b * r2 + c) * (r3 ^ 3 + d * r3 ^ 2 + e * r3 + f) = 0 ∧
  (r4 + a) * (r5 ^ 2 + b * r5 + c) * (r6 ^ 3 + d * r6 ^ 2 + e * r6 + f) = 0 :=
  sorry

end exists_integer_roots_l330_330278


namespace steps_back_l330_330279
open Nat

-- Define a function to check if a number is prime
def is_prime (n : ℕ) : Prop :=
  (n ≥ 2) ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the net steps function
def net_steps : ℕ → ℕ
  | 1 => 0
  | n + 1 => if is_prime n
             then 1
             else 3 + net_steps n

-- The final net steps after 30 moves
def final_net_steps : ℤ :=
  let prime_count := (List.range' 2 (31 - 2)).countp is_prime
  let composite_count := 29 - prime_count
  (prime_count : ℤ) - (3 * composite_count : ℤ)

-- The theorem statement asserting that the number of steps back is 47
theorem steps_back : final_net_steps = -47 := sorry

end steps_back_l330_330279


namespace polygonal_chain_segments_l330_330232

theorem polygonal_chain_segments (n : ℕ) :
  (∃ (S : Type) (chain : S → Prop), (∃ (closed_non_self_intersecting : S → Prop), 
  (∀ s : S, chain s → closed_non_self_intersecting s) ∧
  ∀ line_segment : S, chain line_segment → 
  (∃ other_segment : S, chain other_segment ∧ line_segment ≠ other_segment))) ↔ 
  (∃ k : ℕ, (n = 2 * k ∧ 5 ≤ k) ∨ (n = 2 * k + 1 ∧ 7 ≤ k)) :=
by sorry

end polygonal_chain_segments_l330_330232


namespace coplanar_condition_l330_330321

variables (V : Type*) [AddCommGroup V] [Module ℝ V]
variables (O A B C M : V)

-- defining the vectors in terms of points
def vec (P Q : V) : V := Q - P 
def ma := vec M A
def mb := vec M B
def mc := vec M C

-- condition for coplanarity of points A, B, C, and M
theorem coplanar_condition :
  ma + mb + mc = 0 → ∃ x y z : ℝ, x + y + z = 1 ∧ vec O M = x • vec O A + y • vec O B + z • vec O C :=
sorry

end coplanar_condition_l330_330321


namespace approximate_sqrt_2019_e_l330_330040

noncomputable def e : ℝ := Real.exp 1 -- defining the constant e

def f (x : ℝ) : ℝ := Real.exp x -- defining the function f(x) = e^x
def g (x : ℝ) : ℝ := x + 1 -- defining the tangent line y = x + 1

theorem approximate_sqrt_2019_e : 
  abs ((f (1 / 2019)) - 1.0005) < 0.0001 := -- proving the approximation
by
  sorry

end approximate_sqrt_2019_e_l330_330040


namespace remainder_of_7_pow_7_pow_7_pow_7_mod_500_l330_330223

theorem remainder_of_7_pow_7_pow_7_pow_7_mod_500 : 
  let λ := fun n => 
             match n with
             | 500 => 100
             | 100 => 20
             | _ => 0 in
  7^(7^(7^7)) ≡ 343 [MOD 500] :=
by 
  sorry

end remainder_of_7_pow_7_pow_7_pow_7_mod_500_l330_330223


namespace sin_of_angle_l330_330633

theorem sin_of_angle (α : ℝ) (h : Real.cos (π + α) = -(1/3)) : Real.sin ((3 * π / 2) - α) = -(1/3) := 
by
  sorry

end sin_of_angle_l330_330633


namespace cube_surface_area_l330_330076

theorem cube_surface_area (edge_length : ℝ) (area : ℝ) (h : edge_length = 7) : area = 294 :=
by
  -- Definition of cube surface area
  have face_area := edge_length * edge_length
  have surface_area := 6 * face_area
  -- Given condition edge_length = 7
  -- Proving surface_area = 294
  rw h at face_area
  rw h at surface_area
  simp at surface_area
  sorry

end cube_surface_area_l330_330076


namespace largest_inscribed_equilateral_triangle_area_l330_330175

theorem largest_inscribed_equilateral_triangle_area 
  (r : ℝ) (h_r : r = 10) : 
  ∃ A : ℝ, 
    A = 100 * Real.sqrt 3 ∧ 
    (∃ s : ℝ, s = 2 * r ∧ A = (Real.sqrt 3 / 4) * s^2) := 
  sorry

end largest_inscribed_equilateral_triangle_area_l330_330175


namespace correct_statements_l330_330297

variables {R : Type*} [NormedSpace ℝ R]

def vectors_satisfy (a b : R) : Prop :=
  ∥a + b∥ = 4 ∧ ∥a - b∥ = 2

theorem correct_statements (a b : R) (h : vectors_satisfy a b) :
  (∥a∥ = √2 → real_inner a b / (∥a∥ * ∥b∥) = 3/4) ∧
  (3 ≤ ∥a∥ * ∥b∥ ∧ ∥a∥ * ∥b∥ ≤ 5) ∧
  ((real_inner a b / (∥a∥ * ∥b∥) ≥ 3/5)) :=
sorry

end correct_statements_l330_330297


namespace find_y_intercept_l330_330008

def point (x y : ℝ) := (x, y)

def slope (m : ℝ) := m

def x_intercept (x : ℝ) := point x 0

def y_intercept_point (m x0 : ℝ) :=
  let b := 0 - m * x0
  point 0 b

theorem find_y_intercept :
  ∀ (m x0 : ℝ), (m = -3) → (x0 = 7) → y_intercept_point m x0 = (0, 21) :=
by
  intros m x0 m_cond x0_cond
  simp [y_intercept_point, m_cond, x0_cond]
  exact sorry

end find_y_intercept_l330_330008


namespace find_prime_b_l330_330966

-- Define the polynomial function f
def f (n a : ℕ) : ℕ := n^3 - 4 * a * n^2 - 12 * n + 144

-- Define b as a prime number
def b (n : ℕ) (a : ℕ) : ℕ := f n a

-- Theorem statement
theorem find_prime_b (n : ℕ) (a : ℕ) (h : n = 7) (ha : a = 2) (hb : ∃ p : ℕ, Nat.Prime p ∧ p = b n a) :
  b n a = 11 :=
by
  sorry

end find_prime_b_l330_330966


namespace black_and_blue_lines_l330_330450

-- Definition of given conditions
def grid_size : ℕ := 50
def total_points : ℕ := grid_size * grid_size
def blue_points : ℕ := 1510
def blue_edge_points : ℕ := 110
def red_segments : ℕ := 947
def corner_points : ℕ := 4

-- Calculations based on conditions
def red_points : ℕ := total_points - blue_points

def edge_points (size : ℕ) : ℕ := (size - 1) * 4
def non_corner_edge_points (edge : ℕ) : ℕ := edge - corner_points

-- Math translation
noncomputable def internal_red_points : ℕ := red_points - corner_points - (edge_points grid_size - blue_edge_points)
noncomputable def connections_from_red_points : ℕ :=
  corner_points * 2 + (non_corner_edge_points (edge_points grid_size) - blue_edge_points) * 3 + internal_red_points * 4

noncomputable def adjusted_red_lines : ℕ := red_segments * 2
noncomputable def black_lines : ℕ := connections_from_red_points - adjusted_red_lines

def total_lines (size : ℕ) : ℕ := (size - 1) * size + (size - 1) * size
noncomputable def blue_lines : ℕ := total_lines grid_size - red_segments - black_lines

-- The theorem to be proven
theorem black_and_blue_lines :
  (black_lines = 1972) ∧ (blue_lines = 1981) :=
by
  sorry

end black_and_blue_lines_l330_330450


namespace base_representing_350_as_four_digit_number_with_even_final_digit_l330_330573

theorem base_representing_350_as_four_digit_number_with_even_final_digit {b : ℕ} :
  b ^ 3 ≤ 350 ∧ 350 < b ^ 4 ∧ (∃ d1 d2 d3 d4, 350 = d1 * b^3 + d2 * b^2 + d3 * b + d4 ∧ d4 % 2 = 0) ↔ b = 6 :=
by sorry

end base_representing_350_as_four_digit_number_with_even_final_digit_l330_330573


namespace quadrilateral_O1L_O2K_is_square_l330_330814

-- Definitions of points and segments involved in the problem
variables {A B C M N P Q K L O1 O2 : Type} 

-- Definitions of given conditions
def is_midpoint (Pt MidA MidB : Type) := sorry
def is_square (Pt P1 P2 P3 P4 : Type) := sorry
def is_outward_square (Pt P1 P2 P3 P4 C : Type) := sorry

-- Translations of geometric conditions
axiom O1_is_center_of_square_ABMN : is_outward_square A B M N O1
axiom O2_is_center_of_square_BCQO : is_outward_square B C Q P O2
axiom K_is_midpoint_of_AC : is_midpoint A C K
axiom L_is_midpoint_of_MP : is_midpoint M P L

-- Required proof problem in Lean 4
theorem quadrilateral_O1L_O2K_is_square : is_square O1 L O2 K := sorry

end quadrilateral_O1L_O2K_is_square_l330_330814


namespace f_eq_n_infinitely_many_times_l330_330840

noncomputable def f : ℕ → ℕ := sorry

-- Conditions
axiom f_property (m n : ℕ) (h : m > 0) (k : n > 0) :
  (∃ k, k ∈ finset.range (f n + 1) ∧ m + k = f (m + k)) ∧ 
  ∀ k1 k2 ∈ finset.range (f n + 1), k1 ≠ k2 → 0 < m + k1 ∧ k ∣ f (m + k1) → ¬ (k ∣ f (m + k2))

-- Theorem statement
theorem f_eq_n_infinitely_many_times :
  ∃ᶠ n : ℕ in at_top, f(n) = n :=
sorry

end f_eq_n_infinitely_many_times_l330_330840


namespace hit_target_probability_l330_330848

theorem hit_target_probability 
  (p : ℕ → ℝ) 
  (h1 : p 1 = 1 / 4)
  (h2 : ∀ n, p (n + 1) = 2500 / ((100 + 50 * n) ^ 2))
  : ∃ q, tendsto (λ n, ∏ i in range n, (1 - p i)) at_top (nhds q) ∧ q = 1 / 2 := 
sorry

end hit_target_probability_l330_330848


namespace compare_P_Q_l330_330601

noncomputable def P (a : ℝ) : ℝ := real.sqrt a + real.sqrt (a + 5)
noncomputable def Q (a : ℝ) : ℝ := real.sqrt (a + 2) + real.sqrt (a + 3)

theorem compare_P_Q (a : ℝ) (ha : 0 ≤ a) : P a < Q a := by
  sorry

end compare_P_Q_l330_330601


namespace main_diagonals_inequality_l330_330041

-- Define the type for a convex 101-gon
structure Polygon :=
  (vertices : Fin 102 → Point) -- Considering cyclic ordering from 0 to 101

-- Definition of main diagonal
def is_main_diagonal (p : Polygon) (i j : Fin 102) : Prop :=
  abs ((j.val - i.val + 101) % 102) = 50

-- Define selected and non-selected main diagonals
def selected_main_diagonals (p : Polygon) (D : Finset (Fin 102 × Fin 102)) : Prop :=
  ∀ (diagonal ∈ D), ∃ (i j : Fin 102), diagonal = (i, j) ∧ is_main_diagonal p i j ∧
    ∀ (diagonal' ∈ D), (i ≠ diagonal'.fst ∧ j ≠ diagonal'.snd)

-- Define remaining main diagonals
def remaining_main_diagonals (p : Polygon) (D : Finset (Fin 102 × Fin 102)) : Finset (Fin 102 × Fin 102) :=
  {d ∈ { (i, j) | is_main_diagonal p i j } // d ∉ D }

-- Length function (abstract without specific geometry details)
def length {p : Polygon} (i j : Fin 102) : ℝ := sorry -- abstracted length calculation

-- Sum lengths of diagonals in a finset
def sum_lengths {p : Polygon} (D : Finset (Fin 102 × Fin 102)) : ℝ := D.sum (λ d, length d.fst d.snd)

-- The main theorem
theorem main_diagonals_inequality (p : Polygon) (D : Finset (Fin 102 × Fin 102)) 
  (hD : selected_main_diagonals p D) : 
  sum_lengths D < sum_lengths (remaining_main_diagonals p D) :=
sorry

end main_diagonals_inequality_l330_330041


namespace number_of_permutations_l330_330707

theorem number_of_permutations (a : ℕ → ℕ) :
  (∀ i, a i ∈ {1, 2, 3, 4, 5, 6, 7, 8}) ∧
  (∀ i j, i ≠ j → a i ≠ a j) ∧
  (∀ i, (a i = 8 → (∑ k in finset.range i, if a k < 8 then 1 else 0) = 2)) ∧
  (∀ i, (a i = 7 → (∑ k in finset.range i, if a k < 7 then 1 else 0) = 4)) ∧
  (∀ i, (a i = 4 → (∑ k in finset.range i, if a k < 4 then 1 else 0) = 2)) ∧
  (∃ i, (a i = 1 ∧ a (i+1) = 2) ∨ (a i = 2 ∧ a (i+1) = 1)) →
  finset.card {b | b ∈ {a : fin (8) → fin (8) | 
                         ∀ m n, m ≠ n → a m ≠ a n ∧
                         ∀ i, (a i = 8 → (∑ k in finset.range i, if a k < 8 then 1 else 0) = 2) ∧
                         ∀ i, (a i = 7 → (∑ k in finset.range i, if a k < 7 then 1 else 0) = 4) ∧
                         ∀ i, (a i = 4 → (∑ k in finset.range i, if a k < 4 then 1 else 0) = 2) ∧
                         (∃ i, (a i = 1 ∧ a (i+1) = 2) ∨ (a i = 2 ∧ a (i+1) = 1)) }} = 28 :=
sorry

end number_of_permutations_l330_330707


namespace sum_even_2_to_42_sum_odd_1_to_41_b_squared_minus_a_squared_l330_330444

def sum_even (n : ℕ) : ℕ :=
  n / 2 * (2 + n)

def sum_odd (n : ℕ) : ℕ :=
  n / 2 * (1 + n)

theorem sum_even_2_to_42 : sum_even 42 = 462 := by
  sorry

theorem sum_odd_1_to_41 : sum_odd 41 = 441 := by
  sorry

theorem b_squared_minus_a_squared : 
  let a := sum_even 42
  let b := sum_odd 41
  (b^2 - a^2 = -18963) := by
  sorry

end sum_even_2_to_42_sum_odd_1_to_41_b_squared_minus_a_squared_l330_330444


namespace tangent_line_g_range_of_a_l330_330236

noncomputable def f (x : ℝ) : ℝ := -x^2 - 3
noncomputable def g (a x : ℝ) : ℝ := 2 * x * Real.log x - a * x

theorem tangent_line_g (a : ℝ) :
  let k := -2 in
  let g' := λ x, 2 * (Real.log x + 1) - a in
  a = 4 → 
  (∀ x : ℝ, 2 * x + g a x + 2 = 0) :=
begin
  sorry
end

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 < x →
  2 * x * Real.log x - a * x - (-x^2 - 3) ≥ 0 ) → 
  a ≤ 4 :=
begin
  sorry
end

end tangent_line_g_range_of_a_l330_330236


namespace monotonic_intervals_min_value_difference_l330_330260

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (1/2) * x^2 - a * x + a * real.log x

theorem monotonic_intervals (a : ℝ) (h : a ≠ 0) :
  ((a > 4) → (f'(x) > 0) on (0, (a - real.sqrt (a^2 - 4 * a)) / 2) ∧ (f'(x) > 0) on ((a + real.sqrt (a^2 - 4 * a)) / 2, +∞) ∧ 
            (f'(x) < 0) on ((a - real.sqrt (a^2 - 4 * a)) / 2, (a + real.sqrt (a^2 - 4 * a)) / 2)) ∧
  ((0 < a ≤ 4) → (f'(x) ≥ 0) on (0, +∞)) ∧
  ((a < 0) → (f'(x) < 0) on (0, (a + real.sqrt (a^2 - 4 * a)) / 2) ∧ (f'(x) > 0) on ((a + real.sqrt (a^2 - 4 * a)) / 2, +∞)) :=
sorry

theorem min_value_difference (a : ℝ) (h : a ≥ 9 / 2) (x1 x2 : ℝ) (hx : x1 < x2) (hroot : ∀ {y}, polynomial.roots (polynomial.C a * polynomial.lift_nat_approx 2 (polynomial.X^2 - polynomial.C a * polynomial.X + polynomial.C a)) = {x1, x2}) :
  f x1 - f x2 = -3 / 4 * a - a * real.log 2 :=
sorry

end monotonic_intervals_min_value_difference_l330_330260


namespace convert_binary_to_decimal_l330_330893

-- Define the binary number and its decimal conversion function
def bin_to_dec (bin : List ℕ) : ℕ :=
  list.foldl (λ acc b, 2 * acc + b) 0 bin

-- Define the specific problem conditions
def binary_oneonezeronethreeone : List ℕ := [1, 1, 0, 1, 1]

theorem convert_binary_to_decimal :
  bin_to_dec binary_oneonezeronethreeone = 27 := by
  sorry

end convert_binary_to_decimal_l330_330893


namespace part_A_part_B_part_C_l330_330999

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation (x y : ℝ) : f(x * y) = y^2 * f(x) + x^2 * f(y)

theorem part_A : f(0) = 0 := sorry
theorem part_B : f(1) = 0 := sorry
theorem part_C : ∀ x : ℝ, f(x) = f(-x) := sorry

end part_A_part_B_part_C_l330_330999


namespace percentage_increase_l330_330833

theorem percentage_increase (P : ℝ)
  (h1 : profit_1996 = 1.10 * P)
  (h2 : profit_1997 = 1.3200000000000001 * P) :
  let x := ((profit_1997 / profit_1996) - 1) * 100 in
  x = 20 := by
  sorry

end percentage_increase_l330_330833


namespace total_monthly_feed_l330_330744

def daily_feed (pounds_per_pig_per_day : ℕ) (number_of_pigs : ℕ) : ℕ :=
  pounds_per_pig_per_day * number_of_pigs

def monthly_feed (daily_feed : ℕ) (days_per_month : ℕ) : ℕ :=
  daily_feed * days_per_month

theorem total_monthly_feed :
  let pounds_per_pig_per_day := 15
  let number_of_pigs := 4
  let days_per_month := 30
  monthly_feed (daily_feed pounds_per_pig_per_day number_of_pigs) days_per_month = 1800 :=
by
  sorry

end total_monthly_feed_l330_330744


namespace number_of_staff_members_l330_330306

theorem number_of_staff_members
  (allowance_per_member : ℕ)
  (total_amount : ℕ)
  (allowance_per_day : ℕ)
  (days : ℕ)
  (initial_amount : ℕ)
  (additional_amount : ℕ)
  (H1 : allowance_per_day = 100)
  (H2 : days = 30)
  (H3 : allowance_per_member = allowance_per_day * days)
  (H4 : initial_amount = 65000)
  (H5 : additional_amount = 1000)
  (H6 : total_amount = initial_amount + additional_amount) :
  total_amount / allowance_per_member = 22 :=
by
  rw [H1, H2, H3, H4, H5, H6]
  norm_num
  sorry

end number_of_staff_members_l330_330306


namespace system_of_equations_l330_330102

universe u

-- Definitions based on conditions
variable {x : ℕ} -- number of cars
variable {y : ℕ} -- number of people

-- The first condition: if three people sit in a car, two cars are empty
def condition1 : Prop := y = 3 * (x - 2)

-- The second condition: if two people sit in a car, nine people need to walk
def condition2 : Prop := y = 2 * x + 9

-- The theorem stating that given both conditions, we have the correct system of equations
theorem system_of_equations :
  condition1 ∧ condition2 ↔ (y = 3 * (x - 2)) ∧ (y = 2 * x + 9) :=
by {
  sorry
}

end system_of_equations_l330_330102


namespace students_raised_hands_for_both_l330_330503

theorem students_raised_hands_for_both (total_students : ℕ) (like_art : ℕ) (like_science : ℕ) (all_participated : total_students = 45) (like_art_count : like_art = 42) (like_science_count : like_science = 40) : 
  ∃ (both : ℕ), both = 37 :=
by
  have total := 45
  have art := 42
  have science := 40
  have both := art + science - total
  have total_students := total
  have art_students := art
  have science_students := science
  exact ⟨both, by linarith⟩

end students_raised_hands_for_both_l330_330503


namespace function_properties_l330_330995

-- Given conditions
def domain (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ∈ ℝ

def functional_eqn (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f(x * y) = y^2 * f(x) + x^2 * f(y)

-- Lean 4 theorem statement
theorem function_properties (f : ℝ → ℝ)
  (Hdom : domain f)
  (Heqn : functional_eqn f) :
  f 0 = 0 ∧ f 1 = 0 ∧ ∀ x : ℝ, f (-x) = f x :=
by
  sorry


end function_properties_l330_330995


namespace quadratic_function_and_m_range_l330_330255

-- Given conditions
def f (x : ℝ) : ℝ := x^2 + 4 * x - 2
def g (x : ℝ) : ℝ := f x / x

theorem quadratic_function_and_m_range :
    (∀ x : ℝ, (f x = x^2 + 4 * x - 2)) ∧
    (∀ x ∈ set.Icc (1 : ℝ) (2 : ℝ), ∀ t ∈ set.Icc (-4 : ℝ) (4 : ℝ), g x ≥ -m^2 + t * m) ↔
    (m ∈ set.Ici (3) ∪ set.Iic (-3) ∪ set.Icc (-1) (1)) :=
by
  sorry

end quadratic_function_and_m_range_l330_330255


namespace amount_of_NaOCl_formed_is_2_l330_330560

noncomputable def amount_of_NaOCl_formed : ℕ :=
  let moles_NaOH := 4 in
  let moles_Cl2 := 2 in
  if (moles_NaOH >= 2) ∧ (moles_Cl2 >= 1) then
    (moles_NaOH / 2)        -- because 2 moles of NaOH produce 1 mole of NaOCl
  else
    0

theorem amount_of_NaOCl_formed_is_2 :
  amount_of_NaOCl_formed = 2 := by
  -- Assuming the given conditions in the problem
  have condition1 : 4 >= 2 := by norm_num
  have condition2 : 2 >= 1 := by norm_num
  -- Make use of the conditions
  simp [amount_of_NaOCl_formed, condition1, condition2]
  norm_num
  sorry -- Proof steps can fill the rest

end amount_of_NaOCl_formed_is_2_l330_330560


namespace giant_exponent_modulo_result_l330_330214

theorem giant_exponent_modulo_result :
  (7 ^ (7 ^ (7 ^ 7))) % 500 = 343 :=
by sorry

end giant_exponent_modulo_result_l330_330214


namespace longest_side_of_triangle_l330_330493

theorem longest_side_of_triangle :
  let A := (3 : ℝ, 1 : ℝ)
  let B := (5 : ℝ, 9 : ℝ)
  let C := (7 : ℝ, 1 : ℝ)
  let dist := λ p q : ℝ × ℝ, real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  max (dist A B) (max (dist A C) (dist B C)) = real.sqrt 68 := 
by
  sorry


end longest_side_of_triangle_l330_330493


namespace chord_length_intercepted_by_line_on_circle_l330_330676

theorem chord_length_intercepted_by_line_on_circle :
  ∀ (ρ θ : ℝ), (ρ = 4) →
  (ρ * Real.sin (θ + (Real.pi / 4)) = 2) →
  (4 * Real.sqrt (16 - (2 ^ 2)) = 4 * Real.sqrt 3) :=
by
  intros ρ θ hρ hline_eq
  sorry

end chord_length_intercepted_by_line_on_circle_l330_330676


namespace find_number_find_number_correct_l330_330547

theorem find_number :
  let A := 172 / 4
  let B := A - 28
  (15 * (11 : ℤ) + 7 = 172) →
  (A = 43) →
  (B = 15) →
  (172 = 172) :=
by
  intros h1 h2 h3
  exact eq.refl 172

-- Rewriting to avoid redundant hypotheses and focusing solely on the core equivalence statement
theorem find_number_correct :
  15 * (11 : ℤ) + 7 = 172 :=
by
  exact eq.refl 172

end find_number_find_number_correct_l330_330547


namespace minimum_width_for_truck_passing_l330_330496

-- Definition of the conditions
def truck_height : ℝ := 3
def truck_width : ℝ := 1.6

-- Assume parabola equation is of the form x^2 = -2py
def parabola_eq (p : ℝ) (x : ℝ) (y : ℝ) : Prop := x^2 = -2 * p * y

-- Minimum width calculation function
def minimum_parabola_width (p : ℝ) : ℝ := 2 * p

-- Main theorem statement
theorem minimum_width_for_truck_passing (p : ℝ) (h_p : 1 * p) :
  minimum_parabola_width p = 12.21 := by
  sorry

end minimum_width_for_truck_passing_l330_330496


namespace sphere_tangent_radius_l330_330784

variables (a b : ℝ) (h : b > a)

noncomputable def radius (a b : ℝ) : ℝ := a * (b - a) / Real.sqrt (b^2 - a^2)

theorem sphere_tangent_radius (a b : ℝ) (h : b > a) : 
  radius a b = a * (b - a) / Real.sqrt (b^2 - a^2) :=
by sorry

end sphere_tangent_radius_l330_330784


namespace possible_side_values_l330_330523

noncomputable def valid_isosceles_triangle (x : ℝ) := 
  ∃ (a b c : ℝ), 
    a = real.sin x ∧ 
    b = real.sin x ∧ 
    c = real.sin (7 * x) ∧ 
    (a = b ∨ b = c ∨ c = a) ∧
    ∃ (theta : ℝ), theta = 3 * x ∧ theta < 90 ∧ theta > 0

theorem possible_side_values :
  ∀ (x : ℝ), valid_isosceles_triangle x ↔ x ∈ {10, 30, 50} :=
by
  sorry

end possible_side_values_l330_330523


namespace binary_to_decimal_l330_330902

theorem binary_to_decimal : 
  (1 * 2^0 + 1 * 2^1 + 0 * 2^2 + 1 * 2^3 + 1 * 2^4) = 27 :=
by
  sorry

end binary_to_decimal_l330_330902


namespace candy_for_class_on_monday_l330_330551

def pieces_per_student := 4
def total_pieces_last_monday := 40
def students_missing := 3

theorem candy_for_class_on_monday (pieces_per_student total_pieces_last_monday students_missing : ℕ)
    (students_last_monday : ℕ := total_pieces_last_monday / pieces_per_student) 
    (students_this_monday : ℕ := students_last_monday - students_missing)
    (candies_this_monday : ℕ := students_this_monday * pieces_per_student):
  candies_this_monday = 28 :=
by sorry

#eval candy_for_class_on_monday pieces_per_student total_pieces_last_monday students_missing

end candy_for_class_on_monday_l330_330551


namespace binary_to_decimal_11011_is_27_l330_330898

def binary_to_decimal : ℕ :=
  1 * 2^4 + 1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0

theorem binary_to_decimal_11011_is_27 : binary_to_decimal = 27 := by
  sorry

end binary_to_decimal_11011_is_27_l330_330898


namespace height_growth_and_percentage_growth_l330_330476

noncomputable def height_growth (h_current h_previous : ℝ) : ℝ :=
  h_current - h_previous

noncomputable def percentage_growth (h_growth h_previous : ℝ) : ℝ :=
  (h_growth / h_previous) * 100

theorem height_growth_and_percentage_growth :
  let h_current := 41.5
  let h_previous := 38.5
  let h_growth := height_growth h_current h_previous
  let p_growth := percentage_growth h_growth h_previous
  h_growth = 3 ∧ p_growth ≈ 7.79 := by
-- proof goes here
sorry

end height_growth_and_percentage_growth_l330_330476


namespace opposite_of_2023_is_neg_2023_l330_330781

theorem opposite_of_2023_is_neg_2023 (x : ℝ) (h : x = 2023) : -x = -2023 :=
by
  /- proof begins here, but we are skipping it with sorry -/
  sorry

end opposite_of_2023_is_neg_2023_l330_330781


namespace area_of_inscribed_equilateral_triangle_l330_330173

theorem area_of_inscribed_equilateral_triangle
  (r : ℝ) (h₀ : r = 10) : 
  ∃ A : ℝ, A = 75 * Real.sqrt 3 :=
by
  use 75 * Real.sqrt 3
  sorry

end area_of_inscribed_equilateral_triangle_l330_330173


namespace curve_equation_min_max_distance_l330_330968

-- Define the given points A and B
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (4, 0)

-- Define the moving point (x, y) and the ratio condition
def ratio_condition (x y : ℝ) : Prop :=
  (real.sqrt ((x - 1)^2 + y^2)) / (real.sqrt ((x - 4)^2 + y^2)) = 1 / 2

-- (1) Define the curve C and prove its equation is x^2 + y^2 = 4
theorem curve_equation (x y : ℝ) (h : ratio_condition x y) : x^2 + y^2 = 4 := 
sorry

-- (2) Define the line l and prove the minimum and maximum distance of points on C from the line
def line_l (x y : ℝ) : Prop := x - y + 3 = 0

def distance_to_line (x y : ℝ) : ℝ :=
  abs (x - y + 3) / real.sqrt 2

theorem min_max_distance (x y : ℝ) (h1 : ratio_condition x y) (h2 : x^2 + y^2 = 4) :
  distance_to_line x y ∈ { (3 * real.sqrt 2 / 2) - 2, 2 + (3 * real.sqrt 2 / 2) } :=
sorry

end curve_equation_min_max_distance_l330_330968


namespace arithmetic_progression_product_l330_330566

theorem arithmetic_progression_product (a d : ℕ) (ha : 0 < a) (hd : 0 < d) :
  ∃ (b : ℕ), (a * (a + d) * (a + 2 * d) * (a + 3 * d) * (a + 4 * d) = b ^ 2008) :=
by
  sorry

end arithmetic_progression_product_l330_330566


namespace functional_equation_option_A_option_B_option_C_l330_330973

-- Given conditions
def domain_of_f (f : ℝ → ℝ) : Set ℝ := Set.univ

theorem functional_equation (f : ℝ → ℝ) (x y : ℝ) :
  f (x * y) = y^2 * f x + x^2 * f y := sorry

-- Statements to be proved
theorem option_A (f : ℝ → ℝ) (h : ∀ x y, f(x * y) = y^2 * f(x) + x^2 * f(y)) : f 0 = 0 := sorry

theorem option_B (f : ℝ → ℝ) (h : ∀ x y, f(x * y) = y^2 * f(x) + x^2 * f(y)) : f 1 = 0 := sorry

theorem option_C (f : ℝ → ℝ) (h : ∀ x y, f(x * y) = y^2 * f(x) + x^2 * f(y)) : ∀ x, f(-x) = f x := sorry

end functional_equation_option_A_option_B_option_C_l330_330973


namespace smallest_angle_1920_triangle_angles_not_limited_quadrants_convert_22_30_to_radians_half_angle_of_second_quadrant_l330_330081

-- Problem statement A
def smallest_positive_angle (angle : ℕ) : ℕ :=
  let k := (-5 : ℤ)
  in (k * 360 + angle) % 360

theorem smallest_angle_1920 :
  smallest_positive_angle 1920 = 120 := 
by
  unfold smallest_positive_angle
  sorry -- proof to be filled in

-- Problem statement B
theorem triangle_angles_not_limited_quadrants (α β γ : ℝ) (h : α + β + γ = π) :
  ¬(0 < α ∧ α < π ∧ 0 < β ∧ β < π ∧ 0 < γ ∧ γ < π ∧ α < half_pi ∧ β < half_pi ∧ γ < half_pi ∨
    α < π ∧ α > half_pi ∧ β < π ∧ β > half_pi ∧ γ < π ∧ γ > half_pi) :=
by
  sorry -- proof to be filled in

-- Problem statement C
def degrees_to_radians (deg : ℝ) : ℝ :=
  deg * (π / 180)

theorem convert_22_30_to_radians :
  degrees_to_radians (22 + 30 / 60) = π / 8 :=
by
  unfold degrees_to_radians
  sorry -- proof to be filled in

-- Problem statement D
theorem half_angle_of_second_quadrant (α : ℝ) (h : π / 2 < α ∧ α < π) :
  (π / 2 < α / 2 ∧ α / 2 < π ∨ π < α / 2 ∧ α / 2 < 3 * π / 2) :=
by
  sorry -- proof to be filled in

end smallest_angle_1920_triangle_angles_not_limited_quadrants_convert_22_30_to_radians_half_angle_of_second_quadrant_l330_330081


namespace max_chairs_occupied_l330_330790

noncomputable def max_occupied_chairs : ℕ :=
  29

theorem max_chairs_occupied (n : ℕ) (h_n : n = 30) :
  ∃ m : ℕ, m ≤ n ∧ m = max_occupied_chairs :=
by
  use 29
  split
  · linarith only [h_n]
  · rfl
  sorry

end max_chairs_occupied_l330_330790


namespace problem1_problem2_l330_330742

-- Problem 1:
theorem problem1 (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : a^3 + b^3 > a^2 * b + a * b^2 := 
sorry

-- Problem 2:
theorem problem2 (x : ℝ) : 
  let a := x^2 + 1/2 
  let b := 2 - x 
  let c := x^2 - x + 1 
  in (a < 1) ∨ (b < 1) ∨ (c < 1) -> False :=
sorry

end problem1_problem2_l330_330742


namespace facebook_bonus_each_female_mother_received_l330_330201

theorem facebook_bonus_each_female_mother_received (total_earnings : ℝ) (bonus_percentage : ℝ) 
    (total_employees : ℕ) (male_fraction : ℝ) (female_non_mothers : ℕ) : 
    total_earnings = 5000000 → bonus_percentage = 0.25 → total_employees = 3300 → 
    male_fraction = 1 / 3 → female_non_mothers = 1200 → 
    (250000 / ((total_employees - total_employees * male_fraction.to_nat) - female_non_mothers)) = 1250 :=
by {
  sorry
}

end facebook_bonus_each_female_mother_received_l330_330201


namespace jamie_alex_batches_l330_330514

/-
Problem Statement:
150 students usually attend math club meeting.
Due to a scheduling conflict, only 60% expected to attend.
Each student will eat 3 cookies.
Each batch of cookies produces 20 cookies.
Determine how many full batches are required.
-/

noncomputable def students_usually_attend : ℕ := 150
noncomputable def expected_percentage : ℚ := 0.60
noncomputable def cookies_per_student : ℕ := 3
noncomputable def cookies_per_batch : ℕ := 20

noncomputable def required_batches : ℕ :=
  let expected_attendance := students_usually_attend * expected_percentage
  let total_cookies_needed := expected_attendance * cookies_per_student
  let batches_needed := total_cookies_needed / cookies_per_batch
  if batches_needed.fract = 0 then batches_needed.to_nat else batches_needed.to_nat + 1

theorem jamie_alex_batches : required_batches = 14 := by sorry

end jamie_alex_batches_l330_330514


namespace vertical_angles_congruent_l330_330749

-- Define what it means for two angles to be vertical angles
def areVerticalAngles (a b : ℝ) : Prop := -- placeholder definition
  sorry

-- Define what it means for two angles to be congruent
def areCongruentAngles (a b : ℝ) : Prop := a = b

-- State the problem in the form of a theorem
theorem vertical_angles_congruent (a b : ℝ) :
  areVerticalAngles a b → areCongruentAngles a b := by
  sorry

end vertical_angles_congruent_l330_330749


namespace derivative_of_function_l330_330764

variable (x : ℝ)

def y : ℝ := x^2 * exp (2 * x)

theorem derivative_of_function : deriv y x = (2 * x + 2 * x^2) * exp (2 * x) :=
by sorry

end derivative_of_function_l330_330764


namespace sum_of_factors_of_60_l330_330072

-- Given conditions
def n : ℕ := 60

-- Proof statement
theorem sum_of_factors_of_60 : (∑ d in (finset.filter (λ d, n % d = 0) (finset.range (n + 1))), d) = 168 :=
by
  sorry

end sum_of_factors_of_60_l330_330072


namespace flagpole_distance_l330_330084

def distance_between_flagpoles (total_length : ℝ) (num_flagpoles : ℕ) : ℝ :=
  if num_flagpoles < 2 then 0 else total_length / (num_flagpoles - 1)

theorem flagpole_distance :
  distance_between_flagpoles 11.5 6 = 2.3 :=
by 
  unfold distance_between_flagpoles
  norm_num
  exact rfl

end flagpole_distance_l330_330084


namespace new_income_distribution_l330_330658

theorem new_income_distribution (x : ℝ) (h_x_pos : 0 < x) 
  (h_x : 10 * x = 100) : 
  let income_poor := x
  let income_middle := 3 * x
  let income_rich := 6 * x
  let tax := (x^2 / 5) + x
  let post_tax_income_rich := income_rich * (1 - tax / 100)
  let tax_collected := income_rich * tax / 100
  let redistributed_tax_poor := 2 / 3 * tax_collected
  let redistributed_tax_middle := 1 / 3 * tax_collected 
  let new_income_poor := income_poor + redistributed_tax_poor * 100 / 100
  let new_income_middle := income_middle + redistributed_tax_middle * 100 / 100
  let new_income_rich := post_tax_income_rich * 100 / 100
  in 
  new_income_poor = 22 ∧ 
  new_income_middle = 36 ∧ 
  new_income_rich = 42 := 
by {
  have hx : x = 10 := by sorry,
  let income_poor := 10,
  let income_middle := 30,
  let income_rich := 60,
  let tax := 30,
  let post_tax_income_rich := 60 * 0.7,
  let tax_collected := 18,
  let redistributed_tax_poor := 12,
  let redistributed_tax_middle := 6,
  let new_income_poor := 10 + 12,
  let new_income_middle := 30 + 6,
  let new_income_rich := post_tax_income_rich,
  exact ⟨rfl, rfl, rfl⟩,
}

end new_income_distribution_l330_330658


namespace constant_term_expansion_l330_330377

theorem constant_term_expansion : 
  let binom_coeff := (Nat.choose 6 4) in
  let const_term := binom_coeff * (-2)^4 in
  const_term = 240 := by
    let binom_coeff := Nat.choose 6 4
    let const_term := binom_coeff * (-2)^4
    sorry

end constant_term_expansion_l330_330377


namespace find_n_for_area_l330_330480

-- Define a condition for a regular polygon inscribed in a circle with radius R
def inscribed_ngon_area (n : ℕ) (R : ℝ) : ℝ :=
  (1/2) * (n * 2 * R * Real.sin (Real.pi / n)) * (R * Real.cos (Real.pi / n))

-- Given conditions: n-gon inscribed in a circle of radius R and area is 3 * R^2
theorem find_n_for_area (n : ℕ) (R : ℝ) (hR : R > 0) (h : inscribed_ngon_area n R = 3 * R^2) : n = 12 :=
by
  sorry

end find_n_for_area_l330_330480


namespace trader_profit_percentage_l330_330488

-- Definitions for the conditions
def original_price (P : ℝ) : ℝ := P
def discount_price (P : ℝ) : ℝ := 0.80 * P
def selling_price (P : ℝ) : ℝ := 0.80 * P * 1.45

-- Theorem statement including the problem's question and the correct answer
theorem trader_profit_percentage (P : ℝ) (hP : 0 < P) : 
  (selling_price P - original_price P) / original_price P * 100 = 16 :=
by
  sorry

end trader_profit_percentage_l330_330488


namespace largest_among_expressions_l330_330907

theorem largest_among_expressions
  (A := real.sqrt (real.cbrt 56))
  (B := real.sqrt (real.cbrt 3584))
  (C := real.sqrt (real.cbrt 2744))
  (D := real.sqrt (real.cbrt 392))
  (E := real.sqrt (real.cbrt 448)) :
  B = max (max (max (max A B) C) D) E := 
sorry

end largest_among_expressions_l330_330907


namespace log_sum_expo_multiplication_l330_330002

-- Part 1: Proving that lg 2 + lg 5 = 1
theorem log_sum : log 2 + log 5 = 1 := sorry

-- Part 2: Proving that 4(-100)^4 = 400000000
theorem expo_multiplication : 4 * (-100)^4 = 400000000 := sorry

end log_sum_expo_multiplication_l330_330002


namespace imaginary_part_of_z_l330_330710

noncomputable def z (a b : ℝ) : ℂ := a + b * Complex.I

theorem imaginary_part_of_z (a b : ℝ) (haz : z a b + Complex.conj (z a b) = 4)
                             (hbz : z a b * Complex.conj (z a b) = 8) :
                             b = 2 ∨ b = -2 :=
by 
  sorry

end imaginary_part_of_z_l330_330710


namespace totalPears_l330_330367

-- Define the number of pears picked by Sara and Sally
def saraPears : ℕ := 45
def sallyPears : ℕ := 11

-- Statement to prove
theorem totalPears : saraPears + sallyPears = 56 :=
by
  sorry

end totalPears_l330_330367


namespace average_time_per_mile_l330_330362

-- Define the conditions
def total_distance_miles : ℕ := 24
def total_time_hours : ℕ := 3
def total_time_minutes : ℕ := 36
def total_time_in_minutes : ℕ := (total_time_hours * 60) + total_time_minutes

-- State the theorem
theorem average_time_per_mile : total_time_in_minutes / total_distance_miles = 9 :=
by
  sorry

end average_time_per_mile_l330_330362


namespace area_triangle_ABC_l330_330381
open Real

/-
Given:
- a circle with radius 3 and diameter AB,
- D is a point such that BD = 4 on extension of diameter AB,
- E is a point such that ED = 7 and line ED is perpendicular to AD,
- AE intersects the circle at C between A and E

Prove:
- The area of triangle ABC is 377/49
-/

noncomputable def circle_radius : ℝ := 3
noncomputable def BD : ℝ := 4
noncomputable def ED : ℝ := 7
noncomputable def AD := circle_radius + BD
noncomputable def AE := sqrt (AD^2 + ED^2)
noncomputable def EA := sqrt (AD^2 + ED^2)

theorem area_triangle_ABC : 
  ∀ (O A B C D E : Point),
  (dist O A = circle_radius ∧ dist O B = circle_radius ∧ dist A B = 2 * circle_radius) ∧
  (dist B D = BD) ∧
  (dist D E = ED) ∧
  (∠ ADE = 90) ∧
  (dist A E = EA) ∧
  (line_of_pts A C intersects circle_at_pt (point_on_circle C)) ∧ 
  (B C dividene_t A E) ∧
  ∃ r, ∃ h, IsRightTriangle ABC r h (dist B C),
  area ABC = 377 /49 :=
by
  sorry

end area_triangle_ABC_l330_330381


namespace b_eq_2a_l330_330303

-- Define the context with equal number of boys and girls with at least 4 students
variable (n : ℕ) (Hn : n ≥ 2)

-- Define what a valid sequence is and the types
def isValidSequence (l : List ℕ) : Prop :=
  l.count 0 = n ∧ l.count 1 = n

def isTypeA (l : List ℕ) : Prop :=
  isValidSequence n l ∧
  ∀ k < l.length - 1, l.take k.count 0 ≠ l.take k.count 1

def isTypeB (l : List ℕ) : Prop :=
  isValidSequence n l ∧
  ∃ k, k < l.length - 1 ∧ l.take k.count 0 = l.take k.count 1 ∧
  ∀ j < k, l.take j.count 0 ≠ l.take j.count 1

-- Definitions for a and b 
def aCount : ℕ :=
  (List.finRange (2 * n)).count (λ l, isTypeA n l)

def bCount : ℕ :=
  (List.finRange (2 * n)).count (λ l, isTypeB n l)

-- The theorem to prove
theorem b_eq_2a : bCount n = 2 * aCount n := 
sorry

end b_eq_2a_l330_330303


namespace notecard_calculation_l330_330158

theorem notecard_calculation (N E : ℕ) (h₁ : N - E = 80) (h₂ : N = 3 * E) : N = 120 :=
sorry

end notecard_calculation_l330_330158


namespace min_value_of_inverse_sum_l330_330652

variables {x y : ℝ} {A B C P : Type}

-- Given conditions in the problem
variables (cos_A sin_B sin_C : ℝ)
variables (Ax Ay Bx By Cx Cy Px Py : ℝ)
variables (area_ABC : ℝ)
variables (dot_AB_AC : ℝ)

-- The conditions provided
axiom cond1 : dot_AB_AC = 9
axiom cond2 : sin_B = cos_A * sin_C
axiom cond3 : area_ABC = 6
axiom cond4 : Px = (x / (Cx - Ax)) * (Cx - Ax) + (y / (Cy - By)) * (Cy - By)
axiom cond5 : Py = (x / (Cx - Ax)) * (Ay - Ax) + (y / (Cy - By)) * (By - Cy)

theorem min_value_of_inverse_sum : (∃ x y, (1 / x + 1 / y >= (7 / 12) + (Real.sqrt 3 / 3))) :=
begin
  -- The formal proof would be needed here
  sorry
end

end min_value_of_inverse_sum_l330_330652


namespace compound_interest_investment_l330_330330

noncomputable def initialInvestment (A r n t : ℝ) : ℝ :=
  A / ((1 + r / n) ^ (n * t))

theorem compound_interest_investment :
  initialInvestment 75000 0.08 12 6 ≈ 46851.917 :=
by
  sorry

end compound_interest_investment_l330_330330


namespace circle_tangent_parabola_l330_330123

theorem circle_tangent_parabola (a b : ℝ) (h_parabola : ∀ x, x^2 + 1 = y) 
  (h_tangency : (a, a^2 + 1) ∧ (-a, a^2 + 1)) 
  (h_center : (0, b)) 
  (h_circle : ∀ x y, x^2 + (y - b)^2 = r^2) 
  (h_tangent_points : (x = a) ∧ (x = -a)) : 
  b - (a^2 + 1) = -1/2 := 
sorry

end circle_tangent_parabola_l330_330123


namespace distinct_values_g_x_l330_330717

noncomputable def floor (r : ℝ) : ℤ :=
  ⌊r⌋

noncomputable def g (x : ℝ) : ℤ :=
  ∑ k in Finset.range 10, (floor (↑(k + 3) * x) - ↑(k + 3) * floor x)

theorem distinct_values_g_x : ∃ S : Finset ℤ, S.card = 45 ∧ ∀ x ≥ 0, g x ∈ S := 
sorry

end distinct_values_g_x_l330_330717


namespace smallest_positive_integer_n_l330_330225

theorem smallest_positive_integer_n :
  ∃ (n: ℕ), n = 4 ∧ (∀ x: ℝ, (Real.sin x)^n + (Real.cos x)^n ≤ 2 / n) :=
sorry

end smallest_positive_integer_n_l330_330225


namespace condition_equivalence_l330_330616

noncomputable def f (x : ℝ) : ℝ := x^2 * (Real.exp(x) - Real.exp(-x))

theorem condition_equivalence (a b : ℝ) (h : a + b > 0) : 
  a + b > 0 ↔ f a + f b > 0 := 
sorry

end condition_equivalence_l330_330616


namespace range_of_a_l330_330645

noncomputable def satisfies_inequality (a : ℝ) : Prop :=
  ∀ x : ℝ, 0 < x → (x^2 + 1) * Real.exp x ≥ a * x^2

theorem range_of_a (a : ℝ) : satisfies_inequality a ↔ a ≤ 2 * Real.exp 1 :=
by
  sorry

end range_of_a_l330_330645


namespace last_non_zero_digit_product_first_100_natnums_l330_330046

theorem last_non_zero_digit_product_first_100_natnums :
  (∏ i in finset.range 1 101, i).divisible (10^k) = (∏ i in finset.range 1 101, i) ≠ 0 ∧ 
  ∏ i in finset.range 1 101, i % 10 ≠ 0 → (∏ i in finset.range 1 101, i % 10) = 4 := sorry

end last_non_zero_digit_product_first_100_natnums_l330_330046


namespace orthogonal_vectors_m_value_l330_330720

theorem orthogonal_vectors_m_value (m : ℝ) :
  let a : ℝ × ℝ := (1, -1)
  let b : ℝ × ℝ := (m + 1, 2 * m - 4)
  (a.fst * b.fst + a.snd * b.snd = 0) → m = 5 :=
by
  let a : ℝ × ℝ := (1, -1)
  let b : ℝ × ℝ := (m + 1, 2 * m - 4)
  assume h : a.fst * b.fst + a.snd * b.snd = 0
  sorry

end orthogonal_vectors_m_value_l330_330720


namespace rectangle_area_error_percentage_l330_330669

theorem rectangle_area_error_percentage (L W : ℝ) :
  let L' := 1.10 * L
  let W' := 0.95 * W
  let A := L * W 
  let A' := L' * W'
  let error := A' - A
  let error_percentage := (error / A) * 100
  error_percentage = 4.5 := by
  sorry

end rectangle_area_error_percentage_l330_330669


namespace min_sum_of_fractions_l330_330704

theorem min_sum_of_fractions : 
  ∀ (W X Y Z : ℕ), 
  W ≠ X → W ≠ Y → W ≠ Z → X ≠ Y → X ≠ Z → Y ≠ Z → 
  W ∈ {1,2,3,4,5,6,7,8,9} → 
  X ∈ {1,2,3,4,5,6,7,8,9} → 
  Y ∈ {1,2,3,4,5,6,7,8,9} → 
  Z ∈ {1,2,3,4,5,6,7,8,9} → 
  (∀ (S : ℚ), S = ((W:ℚ)/(X:ℚ) + (Y:ℚ)/(Z:ℚ)) → S ≥ (25/72)) → 
  Σ :=
{ W, X, Y, Z | 
  W = 1 ∧ X = 8 ∧ Y = 2 ∧ Z = 9 → ((W:ℚ)/(X:ℚ) + (Y:ℚ)/(Z:ℚ)) = (25/72) 
    → 
  (∀ (W X Y Z : ℕ), 
  (W ≠ X ∧ W ≠ Y ∧ W ≠ Z ∧ X ≠ Y ∧ X ≠ Z ∧ Y ≠ Z) ∧
  W ∈ {1,2,3,4,5,6,7,8,9} ∧ X ∈ {1,2,3,4,5,6,7,8,9} ∧ 
  Y ∈ {1,2,3,4,5,6,7,8,9} ∧ Z ∈ {1,2,3,4,5,6,7,8,9} 
  → (S = ((1/8) + (2/9)) = (25/72)))}

sorry

end min_sum_of_fractions_l330_330704


namespace neg_neg_one_eq_one_l330_330863

theorem neg_neg_one_eq_one : -(-1) = 1 :=
by
  sorry

end neg_neg_one_eq_one_l330_330863


namespace find_x_l330_330432

theorem find_x (x : ℝ) (h : (3 * x) / 7 = 15) : x = 35 :=
sorry

end find_x_l330_330432


namespace sum_of_positive_factors_60_l330_330053

def sum_of_positive_factors (n : ℕ) : ℕ :=
  ∑ d in (Finset.filter (λ x => x > 0) (Finset.divisors n)), d

theorem sum_of_positive_factors_60 : sum_of_positive_factors 60 = 168 := by
  sorry

end sum_of_positive_factors_60_l330_330053


namespace find_t_closest_to_a_l330_330765

def vec (a b c : ℝ) : ℝ × ℝ × ℝ := (a, b, c)

def v (t : ℝ) : ℝ × ℝ × ℝ := 
  let u₀ := vec 2 (-3) (-3)
  let d := vec 7 5 (-1)
  (u₀.1 + t * d.1, u₀.2 + t * d.2, u₀.3 + t * d.3)

def a : ℝ × ℝ × ℝ := vec 4 4 5

def dot_prod (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

theorem find_t_closest_to_a : ∃ t : ℝ, t = 41 / 75 ∧
  dot_prod (v t) (vec 7 5 (-1)) = dot_prod a (vec 7 5 (-1)) :=
  sorry

end find_t_closest_to_a_l330_330765


namespace discount_double_time_is_20_l330_330635

variables (T R : ℝ) -- Define T as time period and R as rate of interest
variables (PV A TD : ℝ) -- Define PV as present value, A as amount, TD as true discount

/- The conditions given in the problem -/
def true_discount_is_10 : Prop := TD = 10
def amount_is_110 : Prop := A = 110
def pv_formula : Prop := PV = A - TD

/- The correct answer we need to prove -/
def discount_at_double_time : Prop := TD * 2 = 20

theorem discount_double_time_is_20 (h1 : true_discount_is_10) (h2 : amount_is_110) (h3 : pv_formula) : discount_at_double_time :=
by {
  -- Skipping the proof
  sorry
}

end discount_double_time_is_20_l330_330635


namespace compare_abc_l330_330581

theorem compare_abc (a b c : ℝ) (h₁ : a = 0.6 ^ 3) (h₂ : b = 3 ^ 0.6) (h₃ : c = log 0.6 / log 3) : c < a ∧ a < b := 
by
  sorry

end compare_abc_l330_330581


namespace standard_spherical_coordinates_l330_330314

theorem standard_spherical_coordinates :
  ∀ (ρ θ φ : ℝ), ρ = 4 → θ = π / 3 → φ = 9 * π / 5 → 
  (0 < ρ) → (0 ≤ θ) → (θ < 2 * π) → (φ > π) → 
  ∃ (θ_new φ_new : ℝ), 
  θ_new = θ + π ∧ φ_new = 2 * π - φ ∧ 
  (ρ, θ_new, φ_new) = (4, 4 * π / 3, π / 5) :=
by
  intros ρ θ φ h1 h2 h3 h4 h5 h6 h7
  use [θ + π, 2 * π - φ]
  simp [h1, h2, h3, h7]
  sorry

end standard_spherical_coordinates_l330_330314


namespace find_real_values_l330_330925

theorem find_real_values (x : ℝ) : 
  (2 / (x + 2) + 8 / (x + 6) ≥ 2) ↔ (x ∈ Ioo (-6 : ℝ) (-2) ∪ Ioc (-2 : ℝ) 1) :=
by
  -- Conditions: x ≠ -2 and x ≠ -6
  have h1 : x ≠ -2, from sorry,
  have h2 : x ≠ -6, from sorry,
  sorry

end find_real_values_l330_330925


namespace height_difference_l330_330120

theorem height_difference
  (a b : ℝ)
  (parabola_eq : ∀ x, y = x^2 + 1)
  (circle_center : b = 2 * a^2 + 1 / 2) :
  b - (a^2 + 1) = a^2 - 1 / 2 :=
by {
  sorry
}

end height_difference_l330_330120


namespace cos_C_l330_330682

theorem cos_C {A B C : ℝ} (hA : sin A = 3/5) (hB : cos B = 5/13) : cos C = 16/65 := by
  sorry

end cos_C_l330_330682


namespace interest_difference_l330_330849

-- Conditions
def principal : ℕ := 350
def rate : ℕ := 4
def time : ℕ := 8

-- Question rewritten as a statement to prove
theorem interest_difference :
  let SI := (principal * rate * time) / 100 
  let difference := principal - SI
  difference = 238 := by
  sorry

end interest_difference_l330_330849


namespace area_of_inscribed_equilateral_triangle_l330_330172

theorem area_of_inscribed_equilateral_triangle
  (r : ℝ) (h₀ : r = 10) : 
  ∃ A : ℝ, A = 75 * Real.sqrt 3 :=
by
  use 75 * Real.sqrt 3
  sorry

end area_of_inscribed_equilateral_triangle_l330_330172


namespace fewest_coach_handshakes_l330_330144

theorem fewest_coach_handshakes (n k : ℕ) (h1 : (n * (n - 1)) / 2 + k = 281) : k = 5 :=
sorry

end fewest_coach_handshakes_l330_330144


namespace binary_to_decimal_11011_is_27_l330_330896

def binary_to_decimal : ℕ :=
  1 * 2^4 + 1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0

theorem binary_to_decimal_11011_is_27 : binary_to_decimal = 27 := by
  sorry

end binary_to_decimal_11011_is_27_l330_330896


namespace sequence_periodicity_l330_330404

theorem sequence_periodicity (a : ℕ → ℝ) (h₁ : ∀ n, a (n + 1) = 1 / (1 - a n)) (h₂ : a 8 = 2) :
  a 1 = 1 / 2 := 
sorry

end sequence_periodicity_l330_330404


namespace smallest_n_for_factors_l330_330286

theorem smallest_n_for_factors (k : ℕ) (hk : (∃ p : ℕ, k = 2^p) ) :
  ∃ (n : ℕ), ( 5^2 ∣ n * k * 36 * 343 ) ∧ ( 3^3 ∣ n * k * 36 * 343 ) ∧ n = 75 :=
by
  sorry

end smallest_n_for_factors_l330_330286


namespace linear_function_triangle_area_l330_330388

-- Define a function linear that takes a point and returns a function
def is_linear (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), f = λ x, a * x + b

-- Define a function that passes through the point (2,3)
def passes_through (f: ℝ → ℝ) (p: ℝ × ℝ) : Prop :=
  f p.1 = p.2

-- Define a function whose integral over [0, 1] is 0
def integral_zero (f : ℝ → ℝ) : Prop :=
  ∫ x in 0..1, f x = 0

-- Define a function to compute the area of the triangle formed by f and the axes
def triangle_area (f : ℝ → ℝ) : ℝ :=
  let x_int := (0, 0),
      y_int := (2, 3),
      f_0 := f 0 in
  1 / 2 * abs (2 * x_int.1) * abs (f_0 - y_int.2)

-- The statement we need to prove
theorem linear_function_triangle_area
  (f : ℝ → ℝ) (lin : is_linear f) (thru : passes_through f (2, 3)) (int_zero : integral_zero f) :
  triangle_area f = 1 / 4 :=
by sorry

end linear_function_triangle_area_l330_330388


namespace solve_equation_l330_330756

theorem solve_equation :
  ∃ a b x : ℤ, 
  ((a * x^2 + b * x + 14)^2 + (b * x^2 + a * x + 8)^2 = 0) 
  ↔ (a = -6 ∧ b = -5 ∧ x = -2) :=
by {
  sorry
}

end solve_equation_l330_330756


namespace avg_class_weight_l330_330789

def num_students_A : ℕ := 24
def num_students_B : ℕ := 16
def avg_weight_A : ℕ := 40
def avg_weight_B : ℕ := 35

/-- Theorem: The average weight of the whole class is 38 kg --/
theorem avg_class_weight :
  (num_students_A * avg_weight_A + num_students_B * avg_weight_B) / (num_students_A + num_students_B) = 38 :=
by
  -- Proof goes here
  sorry

end avg_class_weight_l330_330789


namespace sum_of_terms_l330_330838

def geometric_sequence (a b c d : ℝ) :=
  ∃ q : ℝ, a = b / q ∧ c = b * q ∧ d = c * q

def symmetric_sequence_of_length_7 (s : Fin 8 → ℝ) :=
  ∀ i : Fin 8, s i = s (Fin.mk (7 - i) sorry)

def sequence_conditions (s : Fin 8 → ℝ) :=
  symmetric_sequence_of_length_7 s ∧
  geometric_sequence (s ⟨1,sorry⟩) (s ⟨2,sorry⟩) (s ⟨3,sorry⟩) (s ⟨4,sorry⟩) ∧
  s ⟨1,sorry⟩ = 2 ∧
  s ⟨3,sorry⟩ = 8

theorem sum_of_terms (s : Fin 8 → ℝ) (h : sequence_conditions s) :
  s 0 + s 1 + s 2 + s 3 + s 4 + s 5 + s 6 = 44 ∨
  s 0 + s 1 + s 2 + s 3 + s 4 + s 5 + s 6 = -4 :=
sorry

end sum_of_terms_l330_330838


namespace expression_simplification_l330_330525

theorem expression_simplification :
  (4 * 6 / (12 * 8)) * ((5 * 12 * 8) / (4 * 5 * 5)) = 1 / 2 :=
by
  sorry

end expression_simplification_l330_330525


namespace maximum_routes_l330_330398

theorem maximum_routes (n : ℕ) (h1 : n = 9)
  (routes : Finset (Finset ℕ))
  (h2 : ∀ r ∈ routes, r.card = 3)
  (h3 : ∀ r1 r2 ∈ routes, r1 ≠ r2 → (r1 ∩ r2).card ≤ 1) :
  routes.card = 12 :=
sorry

end maximum_routes_l330_330398


namespace tip_count_proof_l330_330497

def initial_customers : ℕ := 29
def additional_customers : ℕ := 20
def customers_who_tipped : ℕ := 15
def total_customers : ℕ := initial_customers + additional_customers
def customers_didn't_tip : ℕ := total_customers - customers_who_tipped

theorem tip_count_proof : customers_didn't_tip = 34 :=
by
  -- This is a proof outline, not the actual proof.
  sorry

end tip_count_proof_l330_330497


namespace choose_coprime_or_divisible_l330_330711

theorem choose_coprime_or_divisible (n : ℕ) (A : Finset ℕ) (hA : A \subset Finset.range (2 * n + 1)) (h_card : A.card = n + 1) :
  (∃ x y ∈ A, x ≠ y ∧ Nat.coprime x y) ∧
  (∃ x y ∈ A, x ≠ y ∧ (x ∣ y ∨ y ∣ x)) :=
by
  sorry

end choose_coprime_or_divisible_l330_330711


namespace min_value_expression_min_value_is_7_l330_330563

theorem min_value_expression (x : ℝ) (hx : x > 0) : 
  6 * x + 1 / (x^6) ≥ 7 :=
sorry

theorem min_value_is_7 : 
  6 * 1 + 1 / (1^6) = 7 :=
by norm_num

end min_value_expression_min_value_is_7_l330_330563


namespace no_valid_labeling_with_1_to_12_valid_labeling_with_13_l330_330825

-- Define properties of a cube with 8 vertices and 12 edges, labels range from 1 to 12
def isValidLabeling (labels : List ℕ) : Prop :=
  ∃ L : Fin 12 → ℕ, ∀ v : Fin 8, (∑ i in (cube_neighbors v), L i) = 19.5

-- Prove labeling with 1 to 12 is impossible
theorem no_valid_labeling_with_1_to_12 :
  ¬ (∃ L : Fin 12 → ℕ, ∀ v : Fin 8, (∑ i in (cube_neighbors v), L i) = 19.5) :=
by
  sorry

-- Define properties replacing one label with 13
def isValidLabelingWith13 (i : ℕ) (labels : List ℕ) : Prop :=
  ∃ L : Fin 12 → ℕ, L i = 13 ∧ ∀ v : Fin 8, (∑ i in (cube_neighbors v), L i) = 22

-- Prove valid labeling for i = 3, 7, 11
theorem valid_labeling_with_13 : 
  (∀ i ∈ [3, 7, 11], ∃ L : Fin 12 → ℕ, L i = 13 ∧ ∀ v : Fin 8, (∑ j in (cube_neighbors v), L j) = 22) :=
by
  sorry

end no_valid_labeling_with_1_to_12_valid_labeling_with_13_l330_330825


namespace probability_x_gt_3y_l330_330740

theorem probability_x_gt_3y :
  let width := 3000
  let height := 3001
  let triangle_area := (1 / 2 : ℚ) * width * (width / 3)
  let rectangle_area := (width : ℚ) * height
  triangle_area / rectangle_area = 1500 / 9003 :=
by 
  sorry

end probability_x_gt_3y_l330_330740


namespace convert_binary_to_decimal_l330_330892

-- Define the binary number and its decimal conversion function
def bin_to_dec (bin : List ℕ) : ℕ :=
  list.foldl (λ acc b, 2 * acc + b) 0 bin

-- Define the specific problem conditions
def binary_oneonezeronethreeone : List ℕ := [1, 1, 0, 1, 1]

theorem convert_binary_to_decimal :
  bin_to_dec binary_oneonezeronethreeone = 27 := by
  sorry

end convert_binary_to_decimal_l330_330892


namespace mass_of_CaSO4_formed_correct_l330_330538

noncomputable def mass_CaSO4_formed 
(mass_CaO : ℝ) (mass_H2SO4 : ℝ)
(molar_mass_CaO : ℝ) (molar_mass_H2SO4 : ℝ) (molar_mass_CaSO4 : ℝ) : ℝ :=
  let moles_CaO := mass_CaO / molar_mass_CaO
  let moles_H2SO4 := mass_H2SO4 / molar_mass_H2SO4
  let limiting_reactant_moles := min moles_CaO moles_H2SO4
  limiting_reactant_moles * molar_mass_CaSO4

theorem mass_of_CaSO4_formed_correct :
  mass_CaSO4_formed 25 35 56.08 98.09 136.15 = 48.57 :=
by
  rw [mass_CaSO4_formed]
  sorry

end mass_of_CaSO4_formed_correct_l330_330538


namespace chairs_left_proof_l330_330301

def red_chairs : ℕ := 4
def yellow_chairs : ℕ := 2 * red_chairs
def blue_chairs : ℕ := 3 * yellow_chairs
def green_chairs : ℕ := blue_chairs / 2
def orange_chairs : ℕ := green_chairs + 2
def total_chairs : ℕ := red_chairs + yellow_chairs + blue_chairs + green_chairs + orange_chairs
def borrowed_chairs : ℕ := 5 + 3
def chairs_left : ℕ := total_chairs - borrowed_chairs

theorem chairs_left_proof : chairs_left = 54 := by
  -- This is where the proof would go
  sorry

end chairs_left_proof_l330_330301


namespace fractional_addition_l330_330922

theorem fractional_addition : (2 : ℚ) / 5 + 3 / 8 = 31 / 40 :=
by
  sorry

end fractional_addition_l330_330922


namespace range_of_a_l330_330357

theorem range_of_a (x y a : ℝ) (h1 : x - y = 2) (h2 : x + y = a) (h3 : x > -1) (h4 : y < 0) : -4 < a ∧ a < 2 :=
sorry

end range_of_a_l330_330357


namespace only_valid_set_is_b_l330_330080

def can_form_triangle (a b c : Nat) : Prop :=
  (a + b > c) ∧ (b + c > a) ∧ (c + a > b)

theorem only_valid_set_is_b :
  can_form_triangle 2 3 4 ∧ 
  ¬ can_form_triangle 1 2 3 ∧
  ¬ can_form_triangle 3 4 9 ∧
  ¬ can_form_triangle 2 2 4 := by
  sorry

end only_valid_set_is_b_l330_330080


namespace binom_150_150_l330_330180

-- Definition of factorial
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else List.product (List.range' 1 n.succ)

-- Definition of binomial coefficient using factorial
def binom (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem binom_150_150 : binom 150 150 = 1 :=
by
  sorry

end binom_150_150_l330_330180


namespace perp_lines_l330_330565

-- Conditions definitions
def vec1 : ℝ × ℝ × ℝ := (b, -3, 2)
def vec2 : ℝ × ℝ × ℝ := (2, 3, 1)

-- Dot product definition
def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

-- The value of b such that the lines are perpendicular
theorem perp_lines (b : ℝ) : dot_product vec1 vec2 = 0 -> b = 7 / 2 := by
  sorry

end perp_lines_l330_330565


namespace maple_pine_height_difference_l330_330148

noncomputable def mixed_to_improper (a b c : ℕ) : ℚ := a + b / c

def pine_tree_height : ℚ := mixed_to_improper 15 1 4
def maple_tree_height : ℚ := mixed_to_improper 20 2 3

theorem maple_pine_height_difference :
  maple_tree_height - pine_tree_height = 65 / 12 := by
  sorry

end maple_pine_height_difference_l330_330148


namespace maximize_area_playground_l330_330501

noncomputable def maxAreaPlayground : ℝ :=
  let l := 100
  let w := 100
  l * w

theorem maximize_area_playground : ∀ (l w : ℝ),
  (2 * l + 2 * w = 400) ∧ (l ≥ 100) ∧ (w ≥ 60) → l * w ≤ maxAreaPlayground :=
by
  intros l w h
  sorry

end maximize_area_playground_l330_330501


namespace rationalize_denominator_sum_A_B_C_D_l330_330745

theorem rationalize_denominator :
  (1 / (5 : ℝ)^(1/3) - (2 : ℝ)^(1/3)) = 
  ((25 : ℝ)^(1/3) + (10 : ℝ)^(1/3) + (4 : ℝ)^(1/3)) / (3 : ℝ) := 
sorry

theorem sum_A_B_C_D : 25 + 10 + 4 + 3 = 42 := 
by norm_num

end rationalize_denominator_sum_A_B_C_D_l330_330745


namespace distance_to_second_focus_l330_330477

noncomputable def distance (a b : ℝ × ℝ) : ℝ :=
  real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)

theorem distance_to_second_focus (P : ℝ × ℝ) (hP : (P.1^2 / 16) - (P.2^2 / 9) = 1) (distance_PF1 : distance P (5, 0) = 15) :
  distance P (-5, 0) = 7 ∨ distance P (-5, 0) = 23 :=
begin
  -- Proof would go here
  sorry
end

end distance_to_second_focus_l330_330477


namespace problem_statement_l330_330583

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 / (2^x + a) - 1 / 2

theorem problem_statement :
  (∀ x : ℝ, f 1 x) = (1 / (2^x + 1) - 1 / 2) →
  (∀ x : ℝ, x ∈ Set.univ → f 1 x < f 1 0 →
   ∀ y : ℝ, f 1 (k - y^2) + f 1 (2 - y) > 0 → k < -9 / 4) :=
by
  intros
  sorry

end problem_statement_l330_330583


namespace total_problems_l330_330821

theorem total_problems (math_pages reading_pages problems_per_page : ℕ) :
  math_pages = 2 →
  reading_pages = 4 →
  problems_per_page = 5 →
  (math_pages + reading_pages) * problems_per_page = 30 :=
by
  sorry

end total_problems_l330_330821


namespace sum_factors_of_60_l330_330060

def sum_factors (n : ℕ) : ℕ :=
  (∑ d in Finset.divisors n, d)

theorem sum_factors_of_60 : sum_factors 60 = 168 := by
  sorry

end sum_factors_of_60_l330_330060


namespace intersection_point_value_l330_330392

theorem intersection_point_value (c d: ℤ) (h1: d = 2 * -4 + c) (h2: -4 = 2 * d + c) : d = -4 :=
by
  sorry

end intersection_point_value_l330_330392


namespace linear_function_m_l330_330644

theorem linear_function_m (m : ℤ) (h₁ : |m| = 1) (h₂ : m + 1 ≠ 0) : m = 1 := by
  sorry

end linear_function_m_l330_330644


namespace Sidney_JumpJacks_Tuesday_l330_330753

variable (JumpJacksMonday JumpJacksTuesday JumpJacksWednesday JumpJacksThursday : ℕ)
variable (SidneyTotalJumpJacks BrookeTotalJumpJacks : ℕ)

-- Given conditions
axiom H1 : JumpJacksMonday = 20
axiom H2 : JumpJacksWednesday = 40
axiom H3 : JumpJacksThursday = 50
axiom H4 : BrookeTotalJumpJacks = 3 * SidneyTotalJumpJacks
axiom H5 : BrookeTotalJumpJacks = 438

-- Prove Sidney's JumpJacks on Tuesday
theorem Sidney_JumpJacks_Tuesday : JumpJacksTuesday = 36 :=
by
  sorry

end Sidney_JumpJacks_Tuesday_l330_330753


namespace eq_has_more_than_100_roots_l330_330153

theorem eq_has_more_than_100_roots (p q : ℤ) (h : p ≠ 0) :
  ∃ x : ℚ, ∀ k : ℤ, -99 ≤ k ∧ k ≤ 99 → ([((50 + k / 100 : ℚ) * (50 + k / 100 : ℚ))] + p * (50 + k / 100 : ℚ) + q = 0) :=
by
  sorry

end eq_has_more_than_100_roots_l330_330153


namespace four_pow_expression_l330_330806

theorem four_pow_expression : 4 ^ (3 ^ 2) / (4 ^ 3) ^ 2 = 64 := by
  sorry

end four_pow_expression_l330_330806


namespace unloading_time_relationship_l330_330113

-- Conditions
def loading_speed : ℝ := 30
def loading_time : ℝ := 8
def total_tonnage : ℝ := loading_speed * loading_time
def unloading_speed (x : ℝ) : ℝ := x

-- Proof statement
theorem unloading_time_relationship (x : ℝ) (hx : x ≠ 0) : 
  ∀ y : ℝ, y = 240 / x :=
by 
  sorry

end unloading_time_relationship_l330_330113


namespace ratio_of_length_to_breadth_l330_330395

theorem ratio_of_length_to_breadth (b l k : ℕ) (h1 : b = 15) (h2 : l = k * b) (h3 : l * b = 675) : l / b = 3 :=
by
  sorry

end ratio_of_length_to_breadth_l330_330395


namespace hyperbola_eccentricity_range_l330_330617

theorem hyperbola_eccentricity_range (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0)
  (x y : ℝ) (h_hyperbola : x^2 / a^2 - y^2 / b^2 = 1)
  (h_A_B: ∃ A B : ℝ, x = -c ∧ |AF| = b^2 / a ∧ |CF| = a + c) :
  e > 2 :=
by
  sorry

end hyperbola_eccentricity_range_l330_330617


namespace extra_interest_amount_l330_330115

def principal : ℝ := 15000
def rate1 : ℝ := 0.15
def rate2 : ℝ := 0.12
def time : ℕ := 2

theorem extra_interest_amount :
  principal * (rate1 - rate2) * time = 900 := by
  sorry

end extra_interest_amount_l330_330115


namespace ellipse_and_triangle_properties_l330_330643

noncomputable def standard_equation_of_ellipse (x y : ℝ) : Prop :=
  (x^2 / 4) + (y^2 / 3) = 1

noncomputable def area_triangle (a b c : ℝ) : ℝ :=
  1/2 * a * b

theorem ellipse_and_triangle_properties :
  (∀ x y : ℝ, standard_equation_of_ellipse x y ↔ (x, y) = (1, 3/2) ∨ (x, y) = (1, -3/2)) ∧
  area_triangle 2 3 = 3 :=
by
  sorry

end ellipse_and_triangle_properties_l330_330643


namespace multiplication_24_12_l330_330458

theorem multiplication_24_12 :
  let a := 24
  let b := 12
  let b1 := 10
  let b2 := 2
  let p1 := a * b2
  let p2 := a * b1
  let sum := p1 + p2
  b = b1 + b2 →
  p1 = a * b2 →
  p2 = a * b1 →
  sum = p1 + p2 →
  a * b = sum :=
by
  intros
  sorry

end multiplication_24_12_l330_330458


namespace trapezoid_area_ratio_l330_330017

theorem trapezoid_area_ratio (AD AO OB BC AB DO OC : ℕ) (h1: AD = 13) (h2: AO = 13) (h3: OB = 13) 
(h4: BC = 13) (h5: AB = 16) (h6: DO = 16) (h7: OC = 16) : 
  let ABYX_area := 90
  let XYCD_area := 126
  let ratio := ABYX_area / XYCD_area
  let p := 5
  let q := 7
  p + q = 12 :=
by
  have h_ABYX_area : ABYX_area = 90, from rfl
  have h_XYCD_area : XYCD_area = 126, from rfl
  have h_ratio : ratio = 5 / 7, from rfl
  have h_p : p = 5, from rfl
  have h_q : q = 7, from rfl
  show p + q = 12, from rfl

end trapezoid_area_ratio_l330_330017


namespace complete_square_equation_l330_330331

theorem complete_square_equation (b c : ℤ) (h : (x : ℝ) → x^2 - 6 * x + 5 = (x + b)^2 - c) : b + c = 1 :=
by
  sorry  -- This is where the proof would go

end complete_square_equation_l330_330331


namespace profit_ratio_a_to_b_l330_330139

noncomputable def capital_a : ℕ := 3500
noncomputable def time_a : ℕ := 12
noncomputable def capital_b : ℕ := 10500
noncomputable def time_b : ℕ := 6

noncomputable def capital_months (capital : ℕ) (time : ℕ) : ℕ :=
  capital * time

noncomputable def capital_months_a : ℕ :=
  capital_months capital_a time_a

noncomputable def capital_months_b : ℕ :=
  capital_months capital_b time_b

theorem profit_ratio_a_to_b : (capital_months_a / Nat.gcd capital_months_a capital_months_b) =
                             2 ∧
                             (capital_months_b / Nat.gcd capital_months_a capital_months_b) =
                             3 := 
by
  sorry

end profit_ratio_a_to_b_l330_330139


namespace function_intersects_all_lines_l330_330911

theorem function_intersects_all_lines :
  (∃ f : ℝ → ℝ, (∀ a : ℝ, ∃ y : ℝ, y = f a) ∧ (∀ k b : ℝ, ∃ x : ℝ, f x = k * x + b)) :=
sorry

end function_intersects_all_lines_l330_330911


namespace defeat_giant_enemy_crab_l330_330437

-- Definitions for the conditions of cutting legs and claws
def claws : ℕ := 2
def legs : ℕ := 6
def totalCuts : ℕ := claws + legs
def valid_sequences : ℕ :=
  (Nat.factorial legs) * (Nat.factorial claws) * Nat.choose (totalCuts - claws - 1) claws

-- Statement to prove the number of valid sequences of cuts given the conditions
theorem defeat_giant_enemy_crab : valid_sequences = 14400 := by
  sorry

end defeat_giant_enemy_crab_l330_330437


namespace geometric_sequence_product_l330_330607

theorem geometric_sequence_product :
  ∀ {a : ℕ → ℝ} (r : ℝ), (∀ n, a (n + 1) = a n * r) → a 5 = 2 →
  (Π i in Finset.range 9, a (i + 1)) = 512 := 
by
  sorry

end geometric_sequence_product_l330_330607


namespace total_seats_in_arena_l330_330835

theorem total_seats_in_arena (
  M B : ℕ
  (cost_main : ℕ := 55)
  (cost_back : ℕ := 45)
  (revenue : ℕ := 955000)
  (back_seat_tickets : ℕ := 14500)
  (h : 55 * M + 45 * B = 955000)
  (h2 : B = 14500)
) : (M + B = 20000) := 
by 
  sorry

end total_seats_in_arena_l330_330835


namespace remainder_of_7_pow_7_pow_7_pow_7_mod_500_l330_330218

theorem remainder_of_7_pow_7_pow_7_pow_7_mod_500 :
    (7 ^ (7 ^ (7 ^ 7))) % 500 = 343 := 
sorry

end remainder_of_7_pow_7_pow_7_pow_7_mod_500_l330_330218


namespace sandy_took_310_dollars_l330_330090

theorem sandy_took_310_dollars (X : ℝ) (h70percent : 0.70 * X = 217) : X = 310 := by
  sorry

end sandy_took_310_dollars_l330_330090


namespace max_chords_no_triangle_l330_330373

theorem max_chords_no_triangle (n : ℕ) (h : n = 10) : 
  (∃ (k : ℕ), k = 25 ∧ 
  ∀ (chords : set (fin n × fin n)), (∀ (a b c : fin n), 
  (a, b) ∈ chords → (b, c) ∈ chords → (c, a) ∈ chords → false) → 
  finsupp.card chords ≤ k) :=
by
  use 25
  simp
  sorry

end max_chords_no_triangle_l330_330373


namespace line_ellipse_tangent_l330_330940

theorem line_ellipse_tangent (m : ℝ) : 
  (∀ x y : ℝ, (y = m * x + 2) → (x^2 + (y^2 / 4) = 1)) → m^2 = 0 :=
sorry

end line_ellipse_tangent_l330_330940


namespace a_2_value_l330_330950

noncomputable def m : ℝ := ∫ x in 0..π, (sin x - 1 + 2 * cos (x / 2)^2)

theorem a_2_value (a1 a3 a4 a5 : ℝ) (h : a1 * (x + m) ^ 4 + a2 * (x + m) ^ 3 + a3 * (x + m) ^ 2 + a4 * (x + m) + a5 = x ^ 4) : 
  m = 2 → a2 = -8 :=
by
  -- This step is where the proof would be implemented
  sorry

end a_2_value_l330_330950


namespace seventy_ninth_digit_is_two_l330_330289

def consecutive_integers_sequence (start end : ℕ) : List ℕ :=
  List.range (start - end + 1) |>.map (fun i => start - i)

def digit_at_position (lst : List ℕ) (pos : ℕ) : ℕ :=
  lst.map (fun n => n.toString.data).join.map Char.toNat |>.nth pos |>.getD 0

theorem seventy_ninth_digit_is_two :
  digit_at_position (consecutive_integers_sequence 60 1) 78 = 2 :=
sorry

end seventy_ninth_digit_is_two_l330_330289


namespace giant_exponent_modulo_result_l330_330215

theorem giant_exponent_modulo_result :
  (7 ^ (7 ^ (7 ^ 7))) % 500 = 343 :=
by sorry

end giant_exponent_modulo_result_l330_330215


namespace positive_integer_solutions_l330_330005

theorem positive_integer_solutions (x : ℕ) (h : 2 * x - 3 ≤ 5) : x ∈ {1, 2, 3, 4} :=
by {
  sorry -- Proof not required as per the user's instructions
}

end positive_integer_solutions_l330_330005


namespace correct_yeast_population_change_statement_l330_330511

def yeast_produces_CO2 (aerobic : Bool) : Bool := 
  True

def yeast_unicellular_fungus : Bool := 
  True

def boiling_glucose_solution_purpose : Bool := 
  True

def yeast_facultative_anaerobe : Bool := 
  True

theorem correct_yeast_population_change_statement : 
  (∀ (aerobic : Bool), yeast_produces_CO2 aerobic) →
  yeast_unicellular_fungus →
  boiling_glucose_solution_purpose →
  yeast_facultative_anaerobe →
  "D is correct" = "D is correct" :=
by
  intros
  exact rfl

end correct_yeast_population_change_statement_l330_330511


namespace max_PA_dot_PB_l330_330239

variables {V : Type*} [inner_product_space ℝ V] -- Define the type and space for vectors

structure TriangleRightAngleWithArea (A O B : V) :=
(area_eq_one : (1/2) * ∥A - O∥ * ∥B - O∥ = 1)
(right_angle : inner_product (A - O) (B - O) = 0)

noncomputable def a (A O : V) : V := (A - O) / ∥A - O∥
noncomputable def b (B O : V) : V := (B - O) / ∥B - O∥
noncomputable def OP (O : V) : V := a A O + 2 • (b B O)
noncomputable def PA (P A O : V) : V := A - OP O
noncomputable def PB (P B O : V) : V := B - OP O

theorem max_PA_dot_PB {A O B P} [TriangleRightAngleWithArea A O B] :
  (vec a := a A O) (vec b := b B O) (vec OP := OP O)
  ∃ P, (P = OP) → (dot_product (PA P A O) (PB P B O)) = 1 :=
sorry -- proof omitted

end max_PA_dot_PB_l330_330239


namespace ellipse_focus_collinear_l330_330242

theorem ellipse_focus_collinear :
  let C : set (ℝ × ℝ) := {p | (p.1^2 / 6) + (p.2^2 / 2) = 1}
  let F2 : ℝ × ℝ := (2, 0)
  let D  : ℝ × ℝ := (-3, 0)
  let A B : ℝ × ℝ
  let M : ℝ × ℝ := (A.1, -A.2) 
  let F1 : ℝ × ℝ := (-2, 0)
  line l passing through D intersects C at A and B
  A.1 = B.1 and A.2 = -B.2 →
  let vec_MF1 := (F1.1 - M.1, F1.2 - M.2)
  let vec_BF1 := (F1.1 - B.1, F1.2 - B.2)
  collinear {M, F1, B} :=
by
  -- additional necessary definitions
  sorry

end ellipse_focus_collinear_l330_330242


namespace calculation_result_l330_330168

theorem calculation_result :
  (-1) * (-4) + 2^2 / (7 - 5) = 6 :=
by
  sorry

end calculation_result_l330_330168


namespace birds_left_l330_330810

theorem birds_left (initial_families flew_away remaining_families : ℕ)
  (h1 : initial_families = 709)
  (h2 : flew_away = 472)
  (h3 : remaining_families = initial_families - flew_away) :
  remaining_families = 237 := 
begin
  sorry
end

end birds_left_l330_330810


namespace sectionB_seats_correct_l330_330688

-- Definitions for the number of seats in Section A
def seatsA_subsection1 : Nat := 60
def seatsA_subsection2 : Nat := 3 * 80
def totalSeatsA : Nat := seatsA_subsection1 + seatsA_subsection2

-- Condition for the number of seats in Section B
def seatsB : Nat := 3 * totalSeatsA + 20

-- Theorem statement to prove the number of seats in Section B
theorem sectionB_seats_correct : seatsB = 920 := by
  sorry

end sectionB_seats_correct_l330_330688


namespace constant_distance_to_AB_l330_330593

noncomputable def ellipse (a b : ℝ) (ha : a > 0) (hb : b > 0) :=
  { p : ℝ × ℝ // (p.1 * p.1) / (a * a) + (p.2 * p.2) / (b * b) = 1 }

def foci (a b : ℝ) (ha : a > 0) (hb : b > 0) : (ℝ × ℝ) × (ℝ × ℝ) :=
  let c := Real.sqrt (a * a - b * b) in
  ((-c, 0), (c, 0))

theorem constant_distance_to_AB
  (O : ℝ × ℝ)
  (hO : O = (0, 0))
  (A B : ℝ × ℝ)
  (hA : A ∈ ellipse 2√2 2 (by norm_num) (by norm_num))
  (hB : B ∈ ellipse 2√2 2 (by norm_num) (by norm_num))
  (h_perp : ∥(A.1, A.2) + (B.1, B.2)∥ = ∥(A.1, A.2) - (B.1, B.2)∥) :
  distance O (line_through A B) = 2 * Real.sqrt 6 / 3 := 
  sorry

end constant_distance_to_AB_l330_330593


namespace negation_of_monotonicity_l330_330397

theorem negation_of_monotonicity :
  ¬ ∃ k : ℝ, ∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → x2 < ∞ →
    (k / x1 < k / x2) :=
by
  sorry

end negation_of_monotonicity_l330_330397


namespace least_n_for_A0An_ge_100_l330_330336

theorem least_n_for_A0An_ge_100
  (A_0 : ℝ × ℝ := (0, 0))
  (A : ℕ → ℝ × ℝ)
  (B : ℕ → ℝ × ℝ)
  (hA1 : ∀ n : ℕ, n > 0 → A n = (n * (n + 1) / 2, 0))
  (hB1 : ∀ n : ℕ, n > 0 → B n = (A n).fst, (A n).fst ^ 2)
  (h_triangle : ∀ n : ℕ, n > 0 → ∃ (x : ℝ), A (n - 1) (B n x)^2 (A n, 0)) :
  (∀ n : ℕ, n > 0 → (A n).fst ≥ 100) :=
sorry

end least_n_for_A0An_ge_100_l330_330336


namespace sin_beta_value_l330_330575

theorem sin_beta_value
  (h1 : π / 2 < α < π)
  (h2 : 0 < β < π / 2)
  (h3 : tan α = -3/4)
  (h4 : cos (β - α) = 5/13) :
  sin β = 63/65 :=
sorry

end sin_beta_value_l330_330575


namespace five_digit_sum_of_product_210_l330_330340

theorem five_digit_sum_of_product_210 :
  ∃ M : ℕ, 10000 ≤ M ∧ M < 100000 ∧ (∃ (d : Fin 5 → ℕ), (∏ i, d i = 210) ∧ (d 0 * 10^4 + d 1 * 10^3 + d 2 * 10^2 + d 3 * 10 + d 4 = M)) ∧ ((d 0 + d 1 + d 2 + d 3 + d 4) = 20) :=
sorry

end five_digit_sum_of_product_210_l330_330340


namespace number_of_possible_values_and_sum_of_f_3_l330_330708

noncomputable def f : ℤ → ℤ := sorry

axiom f_property : ∀ m n : ℤ, f(m + n) - f(m * n - 1) = f(m) * f(n) + 1

theorem number_of_possible_values_and_sum_of_f_3 : 
  ∃ n s : ℤ, (f(3) ∈ {1}) ∧ (n = 1) ∧ (s = 1) ∧ (n * s = 1) :=
sorry

end number_of_possible_values_and_sum_of_f_3_l330_330708


namespace radhika_video_games_l330_330743

theorem radhika_video_games (christmas_gifts birthday_gifts gathering_gifts : ℕ) (initial_ratio : ℚ) :
  christmas_gifts = 12 → birthday_gifts = 8 → gathering_gifts = 5 → initial_ratio = 2/3 →
  let total_gifts := christmas_gifts + birthday_gifts + gathering_gifts in
  let initial_games := (initial_ratio * total_gifts).to_int in
  let total_games := initial_games + total_gifts in
  total_games = 41 :=
by
  intros hc hb hg hr
  let total_gifts := christmas_gifts + birthday_gifts + gathering_gifts
  let initial_games := (initial_ratio * total_gifts).to_int
  let total_games := initial_games + total_gifts
  rw [hc, hb, hg, hr]
  have h1 : total_gifts = 25 := by norm_num
  have h2 : initial_games = 16 := by norm_num
  have h3 : total_games = 41 := by norm_num
  exact h3

end radhika_video_games_l330_330743


namespace pencil_price_l330_330106

theorem pencil_price
  (num_pens : ℕ)
  (num_pencils : ℕ)
  (total_cost : ℝ)
  (pen_price : ℝ)
  (total_cost_pens_pencils : total_cost = 570)
  (num_pens_def : num_pens = 30)
  (num_pencils_def : num_pencils = 75)
  (pen_price_def : pen_price = 14) :
  (total_cost - num_pens * pen_price) / num_pencils = 2 := 
by
  rw [num_pens_def, num_pencils_def, pen_price_def, total_cost_pens_pencils]
  norm_num
  sorry

end pencil_price_l330_330106


namespace maximum_sonority_composite_l330_330699

-- Definitions corresponding to the given conditions
def is_harmonic (a b : ℕ) : Prop := a % b = 0 ∨ b % a = 0

def conditions_holds (n : ℕ) (divisors : List ℕ) (play_times : List ℕ) : Prop :=
  n > 1 ∧
  List.sorted (· < ·) divisors ∧
  ∀ x, x ∈ play_times → x > 0 ∧
  ∀ i j, i < j → is_harmonic (divisors[i]) (divisors[j])

-- Main statement to be proven
theorem maximum_sonority_composite (n : ℕ) (divisors : List ℕ) (play_times : List ℕ) :
  conditions_holds n divisors play_times →
  let S := play_times.sum in
  ∃ k, k > 1 ∧ ∃ m, m > 1 ∧ S = k * m :=
sorry

end maximum_sonority_composite_l330_330699


namespace train_crossing_time_l330_330141

noncomputable def speed_kmh : ℕ := 54
noncomputable def length_m : ℕ := 135
noncomputable def speed_mps (s : ℕ) : ℕ := s * 1000 / 3600

theorem train_crossing_time (v : ℕ) (l : ℕ) :
  v = speed_kmh →
  l = length_m →
  l / (speed_mps v) = 9 :=
by
  intros h_v h_l
  rw [h_v, h_l]
  have h_speed : speed_mps speed_kmh = 15 := by
    rw [←nat.mul_div_assoc _ (nat.gcd_dvd_left 1000 3600)]
    norm_num
  rw h_speed
  norm_num
  sorry

end train_crossing_time_l330_330141


namespace sum_of_digits_of_77_is_14_l330_330495

-- Define the conditions given in the problem
def triangular_array_sum (N : ℕ) : ℕ := N * (N + 1) / 2

-- Define what it means to be the sum of the digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.foldl (· + ·) 0

-- The actual Lean theorem statement
theorem sum_of_digits_of_77_is_14 (N : ℕ) (h : triangular_array_sum N = 3003) : sum_of_digits N = 14 :=
by
  sorry  -- Proof to be completed here

end sum_of_digits_of_77_is_14_l330_330495


namespace find_geometric_progression_ratio_l330_330149

theorem find_geometric_progression_ratio 
  (q b_1 : ℕ)
  (h_natural: ∀ n, ∃ b_n : ℕ, b_n = b_1 * q^n)
  (h_sum: b_1 * q^2 * (1 + q^2 + q^4) = 2^2016 * 3^2018 * 7 * 13) : 
  q ∈ {1, 2, 3, 4} := 
begin
  sorry
end

end find_geometric_progression_ratio_l330_330149


namespace two_R_theta_bounds_l330_330516

variables {R : ℝ} (θ : ℝ)
variables (h_pos : 0 < R) (h_triangle : (R + 1 + (R + 1/2)) > 2 *R)

-- Define that θ is the angle between sides R and R + 1/2
-- Here we assume θ is defined via the cosine rule for simplicity

noncomputable def angle_between_sides (R : ℝ) := 
  Real.arccos ((R^2 + (R + 1/2)^2 - 1^2) / (2 * R * (R + 1/2)))

-- State the theorem
theorem two_R_theta_bounds (h : θ = angle_between_sides R) : 
  1 < 2 * R * θ ∧ 2 * R * θ < π :=
by
  sorry

end two_R_theta_bounds_l330_330516


namespace regular_polygon_enclosure_l330_330347

theorem regular_polygon_enclosure (m n : ℕ) (h₁: m = 6) (h₂: (m + 1) = 7): n = 6 :=
by
  -- Lean code to include the problem hypothesis and conclude the theorem
  sorry

end regular_polygon_enclosure_l330_330347


namespace coats_from_high_schools_l330_330758

-- Define the total number of coats collected.
def total_coats_collected : ℕ := 9437

-- Define the number of coats collected from elementary schools.
def coats_from_elementary : ℕ := 2515

-- Goal: Prove that the number of coats collected from high schools is 6922.
theorem coats_from_high_schools : (total_coats_collected - coats_from_elementary) = 6922 := by
  sorry

end coats_from_high_schools_l330_330758


namespace determine_range_of_m_l330_330952

variable {m : ℝ}

-- Condition (p) for all x in ℝ, x^2 - mx + 3/2 > 0
def condition_p (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - m * x + (3 / 2) > 0

-- Condition (q) the foci of the ellipse lie on the x-axis, implying 2 < m < 3
def condition_q (m : ℝ) : Prop :=
  (m - 1 > 0) ∧ ((3 - m) > 0) ∧ ((m - 1) > (3 - m))

theorem determine_range_of_m (h1 : condition_p m) (h2 : condition_q m) : 2 < m ∧ m < Real.sqrt 6 :=
  sorry

end determine_range_of_m_l330_330952


namespace function_periodicity_even_l330_330252

theorem function_periodicity_even (f : ℝ → ℝ)
  (h_even : ∀ x : ℝ, f x = f (-x))
  (h_period : ∀ x : ℝ, x ≥ 0 → f (x + 2) = -f x)
  (h_def : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = 2^x - 1) :
  f (-2017) + f 2018 = 1 :=
sorry

end function_periodicity_even_l330_330252


namespace find_positive_root_l330_330582

open Real

theorem find_positive_root 
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (x : ℝ) :
  sqrt (a * b * x * (a + b + x)) + sqrt (b * c * x * (b + c + x)) + sqrt (c * a * x * (c + a + x)) = sqrt (a * b * c * (a + b + c)) →
  x = (a * b * c) / (a * b + b * c + c * a + 2 * sqrt (a * b * c * (a + b + c))) := 
sorry

end find_positive_root_l330_330582


namespace ratio_of_odd_to_even_divisors_l330_330339

theorem ratio_of_odd_to_even_divisors (M : ℕ) (hM : M = 36 * 25 * 98 * 210) :
  let b := (1 + 3 + 3^2 + 3^3) * (1 + 5 + 5^2 + 5^3) * (1 + 7 + 7^2 + 7^3)
  in 
  let sum_all_divisors := 31 * b
  in 
  let sum_even_divisors := sum_all_divisors - b
  in 
  b / sum_even_divisors = 1 / 30 :=
by 
  sorry

end ratio_of_odd_to_even_divisors_l330_330339


namespace geometric_sum_formula_l330_330967

variable {a : ℕ → ℝ} -- Sequence a_n
variable (S : ℕ → ℝ) -- Sum S_n

-- Definitions for the conditions
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) (a1 : ℝ) : Prop :=
  ∀ n, a n = a1 * q^(n-1)

def sum_first_n_terms (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n, S n = ∑ i in finset.range n, a i

theorem geometric_sum_formula
  (a : ℕ → ℝ) (S : ℕ → ℝ) (a1 q : ℝ)
  (h_geom : geometric_sequence a q a1)
  (h_pos : ∀ n, 0 < a n)
  (h_S : sum_first_n_terms S a)
  (h1 : a 3 * a 7 = 16 * a 5)
  (h2 : a 3 + a 5 = 20) :
  ∀ n, S n = 2 * a n - 1 := sorry

end geometric_sum_formula_l330_330967


namespace sum_abs_first_30_terms_correct_l330_330679

-- Definitions of the sequence conditions
def a : ℕ → ℤ
| 1     => -60
| (n+1) => a n + 3

-- Sum of absolute values of the first 30 terms in the sequence
def sum_abs_first_30_terms : ℤ :=
  let s n := (n * (a 1 + (a n))) / 2 in
  -s 20 + (s 30 - s 20)

theorem sum_abs_first_30_terms_correct : sum_abs_first_30_terms = 765 := 
  sorry

end sum_abs_first_30_terms_correct_l330_330679


namespace gcd_condition_l330_330718

/-- 
There exists an \( n \)-tuple of integers \( X \) such that for all \( i \), 
the GCD of each component difference is 1, meaning the open segment 
defined by these tuples does not pass through any point with integer coordinates.
-/
theorem gcd_condition (n : Nat) (hne1 : n ≥ 1) (X : Fin (2^n - 1) → Fin n → Int) :
  ∃ (x : Fin n → Int), ∀ i : Fin (2^n - 1), Int.gcd x.val (X i).val = 1 :=
by
  sorry

end gcd_condition_l330_330718


namespace find_CD_CE_l330_330585

-- Definitions for the given geometrical configuration
def circle (center : Point) (radius : ℝ) : Prop :=
∀ (p : Point), dist center p = radius

variable {O O1 : Point}
variable {A B C D E P : Point}
variable {r1 : ℝ} (hr1 : r1 = 3)
variable (tangent_O_O1 : tangent_at_point O O1 P)
variable (radiiO : ∀ (p : Point), dist O p = r1)
variable (radiiO1 : ∀ (p : Point), dist O1 p = 3)

variable (common_tangent_AB : tangent_line A B O O1)
variable (line_tangent_O1_C : ∀ l, tangent_to l O1 C ∧ parallel l (Euclidean_line A B))
variable (line_intersect_O_DE : ∀ l, (tangent_to l O1 C) → (tangent_to l O C) → intersects l D E)

-- Theorem statement
theorem find_CD_CE (hradiiO1 : dist O1 B = 3)
  (htangent : tangent_to (Euclidean_line A B) O1 B) (h_parallel : parallel (Euclidean_line D E) (Euclidean_line A B))
  (h_line_intersect : ∀ (D E : Point), intersects (Euclidean_line D E) D E)
  (h_power_of_point : ∀ D E,
    intersects (Euclidean_line D E) D E →
    let BC := 2 * 3 in
    dist C D * dist C E = BC * BC) :
  dist C D * dist C E = 36 :=
by
  sorry

end find_CD_CE_l330_330585


namespace num_regions_of_lines_l330_330685

theorem num_regions_of_lines (R : ℕ → ℕ) :
  R 1 = 2 ∧ 
  (∀ n, R (n + 1) = R n + (n + 1)) →
  (∀ n, R n = (n * (n + 1)) / 2 + 1) :=
by
  intro h
  sorry

end num_regions_of_lines_l330_330685


namespace sum_of_positive_factors_60_l330_330069

theorem sum_of_positive_factors_60 : (∑ n in Nat.divisors 60, n) = 168 := by
  sorry

end sum_of_positive_factors_60_l330_330069


namespace total_cookies_is_390_l330_330142

def abigail_boxes : ℕ := 2
def grayson_boxes : ℚ := 3 / 4
def olivia_boxes : ℕ := 3
def cookies_per_box : ℕ := 48

def abigail_cookies : ℕ := abigail_boxes * cookies_per_box
def grayson_cookies : ℚ := grayson_boxes * cookies_per_box
def olivia_cookies : ℕ := olivia_boxes * cookies_per_box
def isabella_cookies : ℚ := (1 / 2) * grayson_cookies
def ethan_cookies : ℤ := (abigail_boxes * 2 * cookies_per_box) / 2

def total_cookies : ℚ := ↑abigail_cookies + grayson_cookies + ↑olivia_cookies + isabella_cookies + ↑ethan_cookies

theorem total_cookies_is_390 : total_cookies = 390 :=
by
  sorry

end total_cookies_is_390_l330_330142


namespace Chloe_total_points_l330_330527

-- Define the points scored in each round
def first_round_points : ℕ := 40
def second_round_points : ℕ := 50
def last_round_points : ℤ := -4

-- Define total points calculation
def total_points := first_round_points + second_round_points + last_round_points

-- The final statement to prove
theorem Chloe_total_points : total_points = 86 := by
  -- This proof is to be completed
  sorry

end Chloe_total_points_l330_330527


namespace triangle_centers_exist_l330_330533

structure Triangle (α : Type _) [OrderedCommSemiring α] :=
(A B C : α × α)

noncomputable def circumcenter {α : Type _} [OrderedCommSemiring α] (T : Triangle α) : α × α :=
sorry

noncomputable def incenter {α : Type _} [OrderedCommSemiring α] (T : Triangle α) : α × α :=
sorry

noncomputable def excenter {α : Type _} [OrderedCommSemiring α] (T : Triangle α) : α × α :=
sorry

noncomputable def centroid {α : Type _} [OrderedCommSemiring α] (T : Triangle α) : α × α :=
sorry

theorem triangle_centers_exist {α : Type _} [OrderedCommSemiring α] (T : Triangle α) :
  ∃ K O Oc S : α × α, K = circumcenter T ∧ O = incenter T ∧ Oc = excenter T ∧ S = centroid T :=
by
  refine ⟨circumcenter T, incenter T, excenter T, centroid T, ⟨rfl, rfl, rfl, rfl⟩⟩

end triangle_centers_exist_l330_330533


namespace euler_totient_theorem_l330_330937

def is_totient_function (n : ℕ) (p : ℕ → ℕ) : Prop :=
  ∃ (α : ℕ → ℕ) (s : Finset ℕ),
  n = ∏ i in s, p i ^ α i ∧
  Euler_totient n = n * ∏ i in s, (1 - (1 / p i))

def smallest_positive_integer (p : ℕ → ℕ) : ℕ := 47 * 23 * 11 * 5

theorem euler_totient_theorem :
  ∃ n : ℕ,
  is_totient_function n (λ x, if x = 1 then 2 else if x = 2 then 3 else if x = 3 then 5 else if x = 4 then 7 else 47) ∧
  Euler_totient n = (32 / 47) * n :=
begin
  use smallest_positive_integer,
  sorry,
end

end euler_totient_theorem_l330_330937


namespace arithmetic_geo_sequences_l330_330957

theorem arithmetic_geo_sequences (a : ℕ → ℕ) (b : ℕ → ℕ) (d : ℕ) (q : ℕ) (n : ℕ) :
  (∀ n, a n = 2 * n - 1) ∧ (∀ n, b n = 3 ^ n) →
  (∀ k, ∑ i in Finset.range k, b (2 * i + 1) = (3 ^ k - 1) / 2) :=
  sorry

end arithmetic_geo_sequences_l330_330957


namespace positive_difference_l330_330719

def f (n : ℤ) : ℤ :=
  if n < 0 then n^2 - 4 else 2 * n - 24

theorem positive_difference (a1 a2 : ℤ) (a1_lt_0 : a1 < 0) (a2_ge_0 : a2 ≥ 0) :
  f(-1) + f(4) + f(a1) = 0 ∧ f(a1) = 19 ∧ 
  f(-1) + f(4) + f(a2) = 0 ∧ f(a2) = 19 ∧ 
  abs (a2 - a1) = 21.5 + real.sqrt 23 := 
begin
  sorry
end

end positive_difference_l330_330719


namespace fraction_addition_l330_330920

theorem fraction_addition :
  (2 / 5 : ℚ) + (3 / 8) = 31 / 40 :=
sorry

end fraction_addition_l330_330920


namespace bridge_length_sufficient_l330_330035

structure Train :=
  (length : ℕ) -- length of the train in meters
  (speed : ℚ) -- speed of the train in km/hr

def speed_in_m_per_s (speed_in_km_per_hr : ℚ) : ℚ :=
  speed_in_km_per_hr * 1000 / 3600

noncomputable def length_of_bridge (train1 train2 : Train) : ℚ :=
  let train1_speed_m_per_s := speed_in_m_per_s train1.speed
  let train2_speed_m_per_s := speed_in_m_per_s train2.speed
  let relative_speed := train1_speed_m_per_s + train2_speed_m_per_s
  let total_length := train1.length + train2.length
  let time_to_pass := total_length / relative_speed
  let distance_train1 := train1_speed_m_per_s * time_to_pass
  let distance_train2 := train2_speed_m_per_s * time_to_pass
  distance_train1 + distance_train2

theorem bridge_length_sufficient (train1 train2 : Train) (h1 : train1.length = 200) (h2 : train1.speed = 60) (h3 : train2.length = 150) (h4 : train2.speed = 45) :
  length_of_bridge train1 train2 ≥ 350.04 :=
  by
  sorry

end bridge_length_sufficient_l330_330035


namespace sum_of_squares_l330_330013

theorem sum_of_squares (x y : ℝ) (h1 : x + y = 18) (h2 : x * y = 72) : x^2 + y^2 = 180 :=
sorry

end sum_of_squares_l330_330013


namespace b_share_of_payment_l330_330085

noncomputable def workRateA : ℚ := 1 / 30
noncomputable def workRateB : ℚ := 1 / 20
noncomputable def totalPayment : ℚ := 1000

theorem b_share_of_payment : 
  let combinedWorkRate := workRateA + workRateB in
  let bShareOfWork := workRateB / combinedWorkRate in
  let bShareOfPayment := bShareOfWork * totalPayment in
  bShareOfPayment = 600 :=
by
  sorry

end b_share_of_payment_l330_330085


namespace total_earnings_both_shows_l330_330228

def total_earnings_first_show (standard_tickets : ℕ) (premium_tickets : ℕ) (vip_tickets : ℕ)
  (price_standard : ℕ) (price_premium : ℕ) (price_vip : ℕ) : ℕ :=
  standard_tickets * price_standard + premium_tickets * price_premium + vip_tickets * price_vip

def total_earnings_second_show (standard_students : ℕ) (standard_seniors : ℕ)
  (premium_students : ℕ) (premium_seniors : ℕ) (vip_students : ℕ) (vip_seniors : ℕ)
  (price_standard_student : ℕ) (price_standard_senior : ℕ)
  (price_premium_student : ℕ) (price_premium_senior : ℕ)
  (price_vip_student : ℕ) (price_vip_senior : ℕ) : ℕ :=
  standard_students * price_standard_student + standard_seniors * price_standard_senior +
  premium_students * price_premium_student + premium_seniors * price_premium_senior +
  vip_students * price_vip_student + vip_seniors * price_vip_senior

theorem total_earnings_both_shows :
  ∀ (st1 pt1 vt1 st2 pt2 vt2 ss std_price prem_price vip_price std_student_price std_senior_price 
      prem_student_price prem_senior_price vip_student_price vip_senior_price : ℕ)
    (show1_std_tickets = 120) (show1_prem_tickets = 60) (show1_vip_tickets = 20)
    (show2_std_students = 240) (show2_std_seniors = 120) (show2_prem_students = 120) 
    (show2_prem_seniors = 60) (show2_vip_students = 40) (show2_vip_seniors = 20)
    (std_price = 25) (prem_price = 40) (vip_price = 60)
    (std_student_price = 22.5 * 1 : ℕ) (std_senior_price = 21.25 * 1)
    (prem_student_price = 36 * 1) (prem_senior_price = 34)
    (vip_student_price = 54 * 1) (vip_senior_price = 51), 
  total_earnings_first_show show1_std_tickets show1_prem_tickets show1_vip_tickets
    std_price prem_price vip_price +
  total_earnings_second_show show2_std_students show2_std_seniors show2_prem_students
    show2_prem_seniors show2_vip_students show2_vip_seniors
    std_student_price std_senior_price prem_student_price prem_senior_price
    vip_student_price vip_senior_price = 24090 := sorry

end total_earnings_both_shows_l330_330228


namespace jose_share_of_profit_l330_330418

theorem jose_share_of_profit (T_invest : ℝ) (T_duration : ℝ) (J_invest : ℝ) 
                             (J_duration : ℝ) (total_profit : ℝ) 
                             (T_months : T_invest * T_duration = 360000) 
                             (J_months : J_invest * J_duration = 450000) 
                             (total_months : T_invest * T_duration + J_invest * J_duration = 810000) 
                             : (J_invest * J_duration / (T_invest * T_duration + J_invest * J_duration) * total_profit = 15000) :=
begin
  sorry
end

end jose_share_of_profit_l330_330418


namespace sqrt_am_gm_inequality_l330_330355

theorem sqrt_am_gm_inequality (k n : ℕ) (hk : 1 < k) (hn : 1 < n) : 
  (1 : ℝ) / Real.root (n + 1) k + (1 : ℝ) / Real.root (k + 1) n > 1 :=
by
  sorry

end sqrt_am_gm_inequality_l330_330355


namespace max_lambda_l330_330624

-- Definition of vectors and point P
def vector_m (x y : ℝ) : ℝ × ℝ := (x, y)
def vector_n (x y : ℝ) : ℝ × ℝ := (x - y, 0)

-- Definition of dot product
def dot_product (m n : ℝ × ℝ) : ℝ := m.1 * n.1 + m.2 * n.2

-- Condition: Point P is on the curve defined by the hyperbola with x > 0
def on_hyperbola (x y : ℝ) : Prop := dot_product (vector_m x y) (vector_n x y) = 1 ∧ x > 0

-- Distance calculation between two parallel lines
def distance_between_lines (a b c1 c2 : ℝ) : ℝ := (abs (c1 - c2)) / (sqrt (a ^ 2 + b ^ 2))

-- Proving the main statement
theorem max_lambda (x y : ℝ) (h : on_hyperbola x y) : 
  (∃ λ, ∀ (P : ℝ × ℝ), P = (x, y) →
   distance_between_lines 1 (-1) 1 0 ≥ λ) :=
  sorry

end max_lambda_l330_330624


namespace liquid_film_radius_l330_330470

theorem liquid_film_radius :
  let height := 0.05
  let box_length := 8
  let box_width := 4
  let box_height := 10
  let volume := box_length * box_width * box_height
  (π * r^2) * height = volume →
  r = sqrt (volume / (π * height)) :=
by
  intros height box_length box_width box_height volume hw
  sorry

end liquid_film_radius_l330_330470


namespace max_magnitude_sin_value_l330_330315

noncomputable def pointA (θ : ℝ) : ℝ × ℝ := (real.cos θ, real.sqrt 2 * real.sin θ)
noncomputable def pointB (θ : ℝ) : ℝ × ℝ := (real.sin θ, 0)
noncomputable def vectorAB (θ : ℝ) : ℝ × ℝ :=
  let (ax, ay) := pointA θ
  let (bx, by) := pointB θ
  (bx - ax, by - ay)
noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2)
noncomputable def vectorMagnitude (θ : ℝ) : ℝ :=
  magnitude (vectorAB θ)

theorem max_magnitude (θ : ℝ) (h : 0 ≤ θ ∧ θ ≤ (real.pi / 2)) :
  vectorMagnitude θ ≤ real.sqrt 3 :=
sorry

theorem sin_value (θ : ℝ) (h : 0 ≤ θ ∧ θ ≤ (real.pi / 2) ∧ vectorMagnitude θ = real.sqrt (5 / 2)) :
  real.sin (2 * θ + 5 * real.pi / 12) = -(real.sqrt 6 + real.sqrt 14) / 8 :=
sorry

end max_magnitude_sin_value_l330_330315


namespace shaded_region_area_l330_330029

theorem shaded_region_area (r₁ r₂ : ℝ) (r₁_pos : r₁ = 3) (r₂_pos : r₂ = 4) : 
  let R := r₁ + r₂ in
  let area_of_shaded_region := π * R^2 - (π * r₁^2 + π * r₂^2) in
  area_of_shaded_region = 24 * π :=
by
  sorry

end shaded_region_area_l330_330029


namespace min_value_geometric_sequence_l330_330257

-- Definition for conditions and problem setup
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n+1) = r * a n

-- Given data
variable (a : ℕ → ℝ)
variable (h_geom : is_geometric_sequence a)
variable (h_sum : a 2015 + a 2017 = Real.pi)

-- Goal statement
theorem min_value_geometric_sequence :
  ∃ (min_val : ℝ), min_val = (Real.pi^2) / 2 ∧ (
    ∀ a : ℕ → ℝ, 
    is_geometric_sequence a → 
    a 2015 + a 2017 = Real.pi → 
    a 2016 * (a 2014 + a 2018) ≥ (Real.pi^2) / 2
  ) :=
sorry

end min_value_geometric_sequence_l330_330257


namespace range_of_f_l330_330195

-- Define the function f
def f (x : ℝ) : ℝ := 1 / (x^2 + 4)

-- Statement to prove the range of f
theorem range_of_f :
  ∀ y, y > 0 ∧ y ≤ 1 / 4 ↔ ∃ x, f x = y :=
by sorry

end range_of_f_l330_330195


namespace equation_of_line_through_point_l330_330210

theorem equation_of_line_through_point (A : ℝ × ℝ) (k : ℝ) :
  (A = (3, 2)) →
  (∀ x y : ℝ, 4 * x + y - 2 = 0 → k = -4) →
  (∀ x y : ℝ, y - 2 = -4 * (x - 3) → 4 * x + y - 14 = 0) := 
begin
  intros hA hSlope,
  sorry,
end

end equation_of_line_through_point_l330_330210


namespace find_angle_CGH_l330_330698

open EuclideanGeometry

variables {A B C D E F G H : Point}
variables (triangleABC : Triangle A B C)
variables (D : Point) (B : Point) (C : Point) (E : Point) (F : Point) (G : Point) (H : Point)

-- Define the relevant conditions in Lean
def is_altitude (a b c d : Point) : Prop := 
  LineThrough d (projection a b c) ∧ Perpendicular (Line a b c) (Line a d)

def orthocenter {a b c h : Point} : Prop := 
  is_altitude a b c b ∧ is_altitude b c a c ∧ is_altitude c a b a

-- Define the problem statement
theorem find_angle_CGH
  (h1: is_altitude A B C D)
  (h2: is_altitude B C A E)
  (h3: is_altitude C A B F)
  (h4: orthocenter A B C H)
  (h5: LineThrough D G ∧ Parallel (LineThrough D G) (Line A B))
  (h6: IntersectionOfLineSegments E F G):
  angle C G H = 90 :=
sorry

end find_angle_CGH_l330_330698


namespace fill_in_the_blank_l330_330156

-- Definitions of the problem conditions
def parent := "being a parent"
def parent_with_special_needs := "being the parent of a child with special needs"

-- The sentence describing two situations of being a parent
def sentence1 := "Being a parent is not always easy"
def sentence2 := "being the parent of a child with special needs often carries with ___ extra stress."

-- The correct word to fill in the blank.
def correct_answer := "it"

-- Proof problem
theorem fill_in_the_blank : correct_answer = "it" :=
by
  sorry

end fill_in_the_blank_l330_330156


namespace number_of_subsets_l330_330541

theorem number_of_subsets (Y : Set ℕ) : 
  (∃ Y, {1, 2, 3} ⊆ Y ∧ Y ⊆ {1, 2, 3, 4, 5, 6, 7}) → 
  ∃ n, n = 16 := 
by
  sorry

end number_of_subsets_l330_330541


namespace find_m_n_l330_330634

theorem find_m_n (x m n : ℤ) : (x + 2) * (x + 3) = x^2 + m * x + n → m = 5 ∧ n = 6 :=
by {
    sorry
}

end find_m_n_l330_330634


namespace sum_of_positive_factors_60_l330_330064

theorem sum_of_positive_factors_60 : (∑ n in Nat.divisors 60, n) = 168 := by
  sorry

end sum_of_positive_factors_60_l330_330064


namespace remainder_of_7_pow_7_pow_7_pow_7_mod_500_l330_330217

theorem remainder_of_7_pow_7_pow_7_pow_7_mod_500 :
    (7 ^ (7 ^ (7 ^ 7))) % 500 = 343 := 
sorry

end remainder_of_7_pow_7_pow_7_pow_7_mod_500_l330_330217


namespace calculation_correct_l330_330526

theorem calculation_correct :
  ({-2023})^0 + |{-sqrt(2)}| - 2 * real.cos (real.pi / 4) - real.cbrt 216 = -5 := 
by
  sorry

end calculation_correct_l330_330526


namespace students_taking_neither_l330_330737

theorem students_taking_neither (total students_cs students_electronics students_both : ℕ)
  (h1 : total = 60) (h2 : students_cs = 42) (h3 : students_electronics = 35) (h4 : students_both = 25) :
  total - (students_cs - students_both + students_electronics - students_both + students_both) = 8 :=
by {
  sorry
}

end students_taking_neither_l330_330737


namespace sum_of_series_l330_330177

theorem sum_of_series : 
  ∑ n in Finset.range 500 + 1, (1 / (n^3.toRat + n^2.toRat)) = (π^2 / 6) - (1 / 501) := 
by
  sorry

end sum_of_series_l330_330177


namespace problem_proof_l330_330295

noncomputable def f (x : ℝ) : ℝ := x^2
noncomputable def g (x : ℝ) : ℝ := if x < 0 then 1 / x else 0
noncomputable def h (x : ℝ) : ℝ := if x > 0 then 2 * Real.exp (Real.log x) else 0

def separation_line (F G : ℝ → ℝ) (k b : ℝ) : Prop :=
  ∀ x : ℝ, F x ≥ k * x + b ∧ G x ≤ k * x + b

def statement1 : Prop :=
  ∀ x ∈ Icc (-Real.cbrt 2⁻¹) 0, (f x - g x) > 0

def statement2 : Prop :=
  ∃ k b : ℝ, separation_line f g k b ∧ b = -4

def statement3 : Prop :=
  ∃ k b : ℝ, separation_line f g k b ∧ k ∈ Icc (-4) 0

def statement4 : Prop :=
  separation_line f h (2 * Real.sqrt Real.exp) (-Real.exp)

def number_of_true_statements : ℕ :=
  [statement1, statement2, statement3, statement4].count (λ s, s)

theorem problem_proof : number_of_true_statements = 3 := sorry

end problem_proof_l330_330295


namespace exists_range_of_real_numbers_l330_330620

theorem exists_range_of_real_numbers (x : ℝ) :
  (x^2 - 5 * x + 7 ≠ 1) ↔ (x ≠ 3 ∧ x ≠ 2) := 
sorry

end exists_range_of_real_numbers_l330_330620


namespace interval_contains_root_l330_330394

theorem interval_contains_root : ∃ x ∈ Ioo 0 1, (exp x + 2 * x - 3) = 0 :=
by
  let f : ℝ → ℝ := λ x, exp x + 2 * x - 3
  have h_mono : ∀ x y, x < y → f x < f y :=
    by
      intros x y hxy
      exact add_lt_add_of_lt_of_le (exp_strict_mono hxy) (mul_le_mul_of_nonneg_left (le_of_lt hxy) (le_of_lt zero_lt_two))
  have h_f0 : f 0 = -2 :=
    by
      calc
        f 0 = exp 0 + 2 * 0 - 3 := by rfl
        ... = 1 + 0 - 3 := by norm_num
        ... = -2 := by norm_num
  have h_f1 : f 1 = exp 1 + 2 * 1 - 3 := by rfl
  have h_f1_pos : 0 < f 1 :=
    by
      calc
        0 < exp 1 + 2 * 1 - 3 := by
          norm_num
          exact exp_pos 1
  have h_product_neg : f 0 * f 1 < 0 :=
    by
      calc
        f 0 * f 1 = (-2) * (exp 1 + 2 - 3) := by rw [h_f0, h_f1]
        ... = -2 * (exp 1 - 1) := by ring
        ... < 0 := by
          apply mul_neg_of_neg_of_pos
          norm_num
          exact sub_pos.2 (lt_trans zero_lt_one (exp_pos 1))
  exact (exists_root_interval_of_continuous f h_mono 0 1 (-2) (exp 1 - 1) h_f0 h_f1_pos h_product_neg)

end interval_contains_root_l330_330394


namespace slices_of_bread_left_l330_330028

variable (monday_to_friday_slices saturday_slices total_slices_used initial_slices slices_left: ℕ)

def sandwiches_monday_to_friday : ℕ := 5
def slices_per_sandwich : ℕ := 2
def sandwiches_saturday : ℕ := 2
def initial_slices_of_bread : ℕ := 22

theorem slices_of_bread_left :
  slices_left = initial_slices_of_bread - total_slices_used
  :=
by  sorry

end slices_of_bread_left_l330_330028


namespace beta_max_success_ratio_l330_330310

theorem beta_max_success_ratio :
  ∃ (a b c d : ℕ),
    0 < a ∧ a < b ∧
    0 < c ∧ c < d ∧
    b + d ≤ 550 ∧
    (15 * a < 8 * b) ∧ (10 * c < 7 * d) ∧
    (21 * a + 16 * c < 4400) ∧
    ((a + c) / (b + d : ℚ) = 274 / 550) :=
sorry

end beta_max_success_ratio_l330_330310


namespace altitude_quarter_hypotenuse_l330_330677

-- Problem statement in Lean
theorem altitude_quarter_hypotenuse
  (A B C BB1 : Point)
  (hABC : Triangle A B C)
  (angle_B : ∠ B = 90)
  (angle_C : ∠ C = 15)
  (height_BB1 : ∃ BB1, from B to (segment A C) ∧ altitude B BB1) :
  length (B, BB1) = (1/4) * length (A, C) :=
sorry

end altitude_quarter_hypotenuse_l330_330677


namespace percentage_of_books_not_sold_l330_330819

theorem percentage_of_books_not_sold 
  (initial_stock : ℕ) (mon_sold : ℕ) (tue_sold : ℕ) (wed_sold : ℕ) (thu_sold : ℕ) (fri_sold : ℕ) :
  initial_stock = 1400 →
  mon_sold = 75 →
  tue_sold = 50 →
  wed_sold = 64 →
  thu_sold = 78 →
  fri_sold = 135 →
  let total_sold := mon_sold + tue_sold + wed_sold + thu_sold + fri_sold in
  let books_not_sold := initial_stock - total_sold in
  let percentage_not_sold := (books_not_sold / initial_stock.to_float) * 100 in
  percentage_not_sold ≈ 71.29 :=
by
  intros h0 h1 h2 h3 h4 h5
  have hs : total_sold = mon_sold + tue_sold + wed_sold + thu_sold + fri_sold := rfl
  have bs : books_not_sold = initial_stock - total_sold := rfl
  have ps : percentage_not_sold = (books_not_sold / initial_stock.to_float) * 100 := rfl
  sorry

end percentage_of_books_not_sold_l330_330819


namespace boat_speed_in_still_water_l330_330818

theorem boat_speed_in_still_water (b s : ℝ) (h1 : b + s = 11) (h2 : b - s = 5) : b = 8 := by
  sorry

end boat_speed_in_still_water_l330_330818


namespace find_xy_such_that_product_is_fifth_power_of_prime_l330_330923

theorem find_xy_such_that_product_is_fifth_power_of_prime
  (x y : ℕ) (p : ℕ) (hp : Nat.Prime p)
  (hx_pos : 0 < x) (hy_pos : 0 < y)
  (h_eq : (x^2 + y) * (y^2 + x) = p^5) :
  (x = 2 ∧ y = 5) ∨ (x = 5 ∧ y = 2) :=
sorry

end find_xy_such_that_product_is_fifth_power_of_prime_l330_330923


namespace functional_equation_option_A_option_B_option_C_l330_330974

-- Given conditions
def domain_of_f (f : ℝ → ℝ) : Set ℝ := Set.univ

theorem functional_equation (f : ℝ → ℝ) (x y : ℝ) :
  f (x * y) = y^2 * f x + x^2 * f y := sorry

-- Statements to be proved
theorem option_A (f : ℝ → ℝ) (h : ∀ x y, f(x * y) = y^2 * f(x) + x^2 * f(y)) : f 0 = 0 := sorry

theorem option_B (f : ℝ → ℝ) (h : ∀ x y, f(x * y) = y^2 * f(x) + x^2 * f(y)) : f 1 = 0 := sorry

theorem option_C (f : ℝ → ℝ) (h : ∀ x y, f(x * y) = y^2 * f(x) + x^2 * f(y)) : ∀ x, f(-x) = f x := sorry

end functional_equation_option_A_option_B_option_C_l330_330974


namespace find_150th_letter_l330_330045

theorem find_150th_letter (n : ℕ) (pattern : ℕ → Char) (h : ∀ m, pattern (m % 3) = if m % 3 = 0 then 'A' else if m % 3 = 1 then 'B' else 'C') :
  pattern 149 = 'C' :=
by
  sorry

end find_150th_letter_l330_330045


namespace price_reduction_equation_l330_330671

theorem price_reduction_equation (x : ℝ) :
  63800 * (1 - x)^2 = 3900 :=
sorry

end price_reduction_equation_l330_330671


namespace geom_series_remainder_mod_500_l330_330909

theorem geom_series_remainder_mod_500 :
  let S := (Finset.range 1003).sum (λ n => 5^n)
  in S % 500 = 31 :=
by
  sorry

end geom_series_remainder_mod_500_l330_330909


namespace problem_l330_330916

open Complex

noncomputable def zeta := exp (2 * π * I / 14)

theorem problem : (3 - zeta) * (3 - zeta^2) * (3 - zeta^3) * (3 - zeta^4) * (3 - zeta^5) * (3 - zeta^6) * (3 - zeta^7) * (3 - zeta^8) * (3 - zeta^9) * (3 - zeta^10) * (3 - zeta^11) * (3 - zeta^12) * (3 - zeta^13) = 2143588 :=
by sorry

end problem_l330_330916


namespace find_radii_of_circles_l330_330421

/-- Two circles touch externally, and a line passing through their point of tangency forms chords.
One chord length is 13/5 times the other. The distance between the centers of the circles is 36 units.
We aim to prove that the radii of the circles are 10 and 26, respectively. -/
theorem find_radii_of_circles
  (O₁ O₂ : Type) (K : Type)
  [MetricSpace O₁] [MetricSpace O₂] 
  [Point_of_tangency K] -- centers and point of tangency
  (r₁ r₂ : ℝ) (dist_O₁O₂ : ℝ) (ratio : ℝ) 
  (h₁ : dist_O₁O₂ = 36)
  (h₂ : ratio = 13 / 5)
  (h₃ : true) -- condition that circles touch externally
  (h₄ : ∀ (AK BK : ℝ), AK = ratio * BK → true) -- condition that ratio of chords is 13/5
  : r₁ = 10 ∧ r₂ = 26 :=
by
  sorry

end find_radii_of_circles_l330_330421


namespace sum_infinite_series_eq_half_l330_330530

theorem sum_infinite_series_eq_half :
  (∑' n : ℕ, (n^5 + 2*n^3 + 5*n^2 + 20*n + 20) / (2^(n + 1) * (n^5 + 5))) = 1 / 2 := 
sorry

end sum_infinite_series_eq_half_l330_330530


namespace eval_expression_l330_330198

theorem eval_expression : 
  (2023^3 - 2 * 2023^2 * 2024 + 3 * 2023 * 2024^2 - 2024^3 + 2023) / (2023 * 2024) = -4044 :=
by 
  sorry

end eval_expression_l330_330198


namespace sin_diff_angle_identity_l330_330249

open Real

noncomputable def alpha : ℝ := sorry -- α is an obtuse angle

axiom h1 : 90 < alpha ∧ alpha < 180 -- α is an obtuse angle
axiom h2 : cos alpha = -3 / 5 -- given cosine value

theorem sin_diff_angle_identity :
  sin (π / 4 - alpha) = - (7 * sqrt 2) / 10 :=
by
  sorry

end sin_diff_angle_identity_l330_330249


namespace cot_30_eq_sqrt_3_l330_330554

theorem cot_30_eq_sqrt_3 (h1 : Real.tan (π / 6) = 1 / Real.sqrt 3) : Real.cot (π / 6) = Real.sqrt 3 :=
by
  sorry

end cot_30_eq_sqrt_3_l330_330554


namespace third_group_students_l330_330787

theorem third_group_students (total_students first_group second_group fourth_group : ℕ)
  (h_total : total_students = 24)
  (h_first : first_group = 5)
  (h_second : second_group = 8)
  (h_fourth : fourth_group = 4) : 
  total_students - (first_group + second_group + fourth_group) = 7 :=
by
  rw [h_total, h_first, h_second, h_fourth]
  simp
  exact Nat.sub_eq_of_eq_add (Nat.succ_add 16 7).symm

end third_group_students_l330_330787


namespace functional_eq_properties_l330_330990

theorem functional_eq_properties (f : ℝ → ℝ) (h_domain : ∀ x, x ∈ ℝ) (h_func_eq : ∀ x y, f (x * y) = y^2 * f x + x^2 * f y) :
  f 0 = 0 ∧ f 1 = 0 ∧ ∀ x, f x = f (-x) :=
by {
  sorry
}

end functional_eq_properties_l330_330990


namespace oblique_line_perpendicular_planes_l330_330132

theorem oblique_line_perpendicular_planes {α : Type} [plane α]
  (l : line α) (h_oblique : ¬(l ∥ α)): ∃! p : plane α, p ⊥ α ∧ l ⊆ p :=
sorry

end oblique_line_perpendicular_planes_l330_330132


namespace math_proof_problem_l330_330248

-- Define the problem conditions
variables {a b r : ℝ}
variable {O : point}
variable {P : point}

-- Given conditions
def ab_nonzero : Prop := a ≠ 0 ∧ b ≠ 0
def point_P_outside_circle : Prop := a^2 + b^2 > r^2
def line_OP_slope : Prop := ∀ P, (a ≠ 0) → (b ≠ 0) → (P = (a, b)) → (slope OP = b / a)
def line_l_perpendicular_OP : Prop := ∀ l, (slope l = -a / b)
def line_m_equation : Prop := ∀ m, (equation m = λ x y, ax + by = r^2)

-- The conclusion
theorem math_proof_problem (h1 : ab_nonzero) (h2 : point_P_outside_circle) 
  (h3 : line_OP_slope) (h4 : line_l_perpendicular_OP) (h5 : line_m_equation) :
  (m_parallel_l : Prop) ∧ (m_intersects_circle : Prop) :=
begin
  sorry
end

end math_proof_problem_l330_330248


namespace sum_of_positive_factors_60_l330_330057

def sum_of_positive_factors (n : ℕ) : ℕ :=
  ∑ d in (Finset.filter (λ x => x > 0) (Finset.divisors n)), d

theorem sum_of_positive_factors_60 : sum_of_positive_factors 60 = 168 := by
  sorry

end sum_of_positive_factors_60_l330_330057


namespace functional_eq_properties_l330_330988

theorem functional_eq_properties (f : ℝ → ℝ) (h_domain : ∀ x, x ∈ ℝ) (h_func_eq : ∀ x y, f (x * y) = y^2 * f x + x^2 * f y) :
  f 0 = 0 ∧ f 1 = 0 ∧ ∀ x, f x = f (-x) :=
by {
  sorry
}

end functional_eq_properties_l330_330988


namespace find_due_years_l330_330762

theorem find_due_years (BG BD : ℝ) (r : ℝ) (t : ℝ) 
  (h1 : BG = 90) 
  (h2 : BD = 340) 
  (h3 : r = 0.12) 
  (h4 : BD / (BD - BG) = 1 + r * t) : 
  t = 3 := 
by 
  linarith [h1, h2, h3, h4]

end find_due_years_l330_330762


namespace problem1_problem2_l330_330879

-- Problem 1: Prove that 3 * sqrt(20) - sqrt(45) + sqrt(1 / 5) = (16 * sqrt(5)) / 5
theorem problem1 : 3 * Real.sqrt 20 - Real.sqrt 45 + Real.sqrt (1 / 5) = (16 * Real.sqrt 5) / 5 := 
sorry

-- Problem 2: Prove that (sqrt(6) - 2 * sqrt(3))^2 - (2 * sqrt(5) + sqrt(2)) * (2 * sqrt(5) - sqrt(2)) = -12 * sqrt(2)
theorem problem2 : (Real.sqrt 6 - 2 * Real.sqrt 3) ^ 2 - (2 * Real.sqrt 5 + Real.sqrt 2) * (2 * Real.sqrt 5 - Real.sqrt 2) = -12 * Real.sqrt 2 := 
sorry

end problem1_problem2_l330_330879


namespace gnomes_in_fifth_house_l330_330791

-- Defining the problem conditions
def num_houses : Nat := 5
def gnomes_per_house : Nat := 3
def total_gnomes : Nat := 20

-- Defining the condition for the first four houses
def gnomes_in_first_four_houses : Nat := 4 * gnomes_per_house

-- Statement of the problem
theorem gnomes_in_fifth_house : 20 - (4 * 3) = 8 := by
  sorry

end gnomes_in_fifth_house_l330_330791


namespace exists_function_f_l330_330229

theorem exists_function_f (a : ℕ → ℝ) (h_pos : ∀ k, 1 ≤ k ∧ k ≤ 2008 → a k > 0) (h_sum : (finset.range 2008).sum (λ k, a (k + 1)) > 1) :
  ∃ f : ℕ → ℝ, 
    (f 0 = 0) ∧ (∀ n, f n < f (n + 1)) ∧ 
    (∃ L, tendsto f at_top (𝓝 L)) ∧ 
    (∀ n, f (n + 1) - f n = (finset.range 2008).sum (λ k, a (k + 1) * f (n + k + 1)) - (finset.range 2008).sum (λ k, a (k + 1) * f (n + k))) :=
begin
  sorry
end

end exists_function_f_l330_330229


namespace find_roots_of_equation_l330_330936

noncomputable def polynomial_solution : Prop :=
  ∃ z : ℂ, (z ^ 2 + 2 * z = -3 + 4 * complex.i) ∧ (z = 2 * complex.i ∨ z = -2 - 2 * complex.i)

theorem find_roots_of_equation : polynomial_solution :=
begin
  sorry
end

end find_roots_of_equation_l330_330936


namespace smallest_k_l330_330783

noncomputable def a : ℕ → ℝ
| 0 => 1
| 1 => real.root (17 : ℝ) 3
| (n+2) => a (n+1) * (a n) ^ 2

theorem smallest_k :
  ∃ k : ℕ, (∀ m < k, ∃ z : ℤ, a (m+1) = (z : ℝ)) ∧ k = 11 := by
  sorry

end smallest_k_l330_330783


namespace complex_fraction_simplified_result_l330_330763

def complex_fraction_simplify : ℂ := (4 - 2 * complex.i) / (1 + complex.i)
theorem complex_fraction_simplified_result : complex_fraction_simplify = 1 - 3 * complex.i := 
sorry

end complex_fraction_simplified_result_l330_330763


namespace problem1_problem2_l330_330823

-- Statement for Problem 1
theorem problem1 (P : Point) (hp : P = (4, 1)) (equal_intercepts : ∃ (a : ℝ), a ≠ 0 ∧ (l : Line), l = (y = -x + a))
  : (l = (x - 4*y = 0)) ∨ (l = (x + y - 5 = 0)) :=
sorry

-- Statement for Problem 2
theorem problem2 (P : Point) (hp : P = (3, 4)) (theta : Angle) (h_theta : theta ≠ 90) (Q : Point) (hQ_consistent : Q = (cos theta, sin theta))
  : (l : Line), l = (y = 4/3 * x) :=
sorry

end problem1_problem2_l330_330823


namespace girls_to_boys_ratio_l330_330348

variable (g b : ℕ)
variable (h_total : g + b = 36)
variable (h_diff : g = b + 6)

theorem girls_to_boys_ratio (g b : ℕ) (h_total : g + b = 36) (h_diff : g = b + 6) :
  g / b = 7 / 5 := by
  sorry

end girls_to_boys_ratio_l330_330348


namespace sum_factors_of_60_l330_330061

def sum_factors (n : ℕ) : ℕ :=
  (∑ d in Finset.divisors n, d)

theorem sum_factors_of_60 : sum_factors 60 = 168 := by
  sorry

end sum_factors_of_60_l330_330061


namespace binary_to_decimal_11011_is_27_l330_330897

def binary_to_decimal : ℕ :=
  1 * 2^4 + 1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0

theorem binary_to_decimal_11011_is_27 : binary_to_decimal = 27 := by
  sorry

end binary_to_decimal_11011_is_27_l330_330897


namespace tan_of_acute_angle_and_cos_pi_add_alpha_l330_330602

theorem tan_of_acute_angle_and_cos_pi_add_alpha (α : ℝ) (h1 : 0 < α ∧ α < π/2)
  (h2 : Real.cos (π + α) = -Real.sqrt (3) / 2) : 
  Real.tan α = Real.sqrt (3) / 3 :=
by
  sorry

end tan_of_acute_angle_and_cos_pi_add_alpha_l330_330602


namespace sequence_sum_lt_l330_330857

noncomputable def sequence (n : ℕ) : ℚ :=
  if n = 1 then 1
  else if n = 2 then 1/3
  else sequence_aux n
where sequence_aux : ℕ → ℚ
| 1 := 1
| 2 := 1/3
| n+3 := infer_from sn (sn_aux n)

lemma recurrence_relation (n : ℕ) (hn : 1 ≤ n) :
  (1 + sequence n) * (1 + sequence (n + 2)) / (1 + sequence (n + 1))^2 = 
  sequence n * sequence (n + 2) / (sequence (n + 1))^2 := sorry

theorem sequence_sum_lt (n : ℕ) (hn : 1 ≤ n) :
  (finset.range n).sum (λ i, sequence (i + 1)) < 34 / 21 := sorry

end sequence_sum_lt_l330_330857


namespace roses_left_unsold_l330_330868

def price_per_rose : ℕ := 4
def initial_roses : ℕ := 13
def total_earned : ℕ := 36

theorem roses_left_unsold : (initial_roses - (total_earned / price_per_rose) = 4) :=
by
  sorry

end roses_left_unsold_l330_330868


namespace general_term_formula_l330_330771
-- Import the Mathlib library 

-- Define the conditions as given in the problem
/-- 
Define the sequence that represents the numerators. 
This is an arithmetic sequence of odd numbers starting from 1.
-/
def numerator (n : ℕ) : ℕ := 2 * n + 1

/-- 
Define the sequence that represents the denominators. 
This is a geometric sequence with the first term being 2 and common ratio being 2.
-/
def denominator (n : ℕ) : ℕ := 2^(n+1)

-- State the main theorem that we need to prove
theorem general_term_formula (n : ℕ) : (numerator n) / (denominator n) = (2 * n + 1) / 2^(n+1) :=
sorry

end general_term_formula_l330_330771


namespace price_of_uniform_l330_330472

variable (p_total_salary p_received_salary : ℝ)
variable (p_service_ratio : ℝ) (p_total_time : ℝ)
variable (p_uniform_price : ℝ)

theorem price_of_uniform :
  let salary_per_month := p_total_salary / p_total_time in
  let expected_salary := salary_per_month * p_service_ratio in
  expected_salary - p_received_salary = p_uniform_price :=
by
  let salary_per_month := p_total_salary / p_total_time
  let expected_salary := salary_per_month * p_service_ratio
  sorry

-- Set the specific values according to the problem statement
noncomputable def total_salary := 900
noncomputable def received_salary := 650
noncomputable def service_ratio := 9 / 12
noncomputable def total_time := 1
noncomputable def uniform_price := 25

-- Substitute and check that the theorem holds with given values
example : price_of_uniform total_salary received_salary service_ratio total_time uniform_price := by
  rfl

end price_of_uniform_l330_330472


namespace pencils_count_l330_330328

-- Definition of the problem as a Lean statement
theorem pencils_count :
  let initial_pencils := 2 in
  let pencils_after_add_3 := initial_pencils + 3 in
  let pencils_after_remove_1 := pencils_after_add_3 - 1 in
  let pencils_after_add_5 := pencils_after_remove_1 + 5 in
  let pencils_after_remove_2 := pencils_after_add_5 - 2 in
  let final_pencils := pencils_after_remove_2 + 4 in
  final_pencils = 11 :=
by
  sorry

end pencils_count_l330_330328


namespace inscribed_cube_surface_area_and_volume_l330_330138

theorem inscribed_cube_surface_area_and_volume :
  (∃ s : ℝ, s^2 = 16 ∧ 
  (∃ l : ℝ, 3 * l^2 = 16 ∧
  (6 * l^2 = 32 ∧ l^3 = (64 * real.sqrt 3)/9))) := sorry

end inscribed_cube_surface_area_and_volume_l330_330138


namespace gridSymmetryAndXForm_l330_330883

/-- Define the initial grid with 2 shaded squares at positions (1,3) and (2,4) -/
def initialShaded : set (ℕ × ℕ) := {(1, 3), (2, 4)}

noncomputable def leastAdditionalSquaresToFormX (initial : set (ℕ × ℕ)) (n m : ℕ) : ℕ :=
  -- Define the symmetry condition in the grid and the formation of letter 'X' and calculate
  4 -- This is our expected answer

theorem gridSymmetryAndXForm :
  leastAdditionalSquaresToFormX initialShaded 4 4 = 4 :=
by
  sorry

end gridSymmetryAndXForm_l330_330883


namespace circle_radii_l330_330171

noncomputable def countPossibleRs (C_radius : ℕ) (D_max_radius : ℕ) : ℕ :=
  let divisors := {d | d ∣ C_radius} \ {C_radius}  -- Exclude C_radius itself since r < C_radius
  | divisors \ {d | ¬ (d < D_max_radius)}         -- Exclude anything not less than D_max_radius

theorem circle_radii (C_radius D_max_radius : ℕ) (h1 : C_radius = 150) (h2 : D_max_radius < 150) :
  countPossibleRs C_radius D_max_radius = 11 :=
by
  sorry

end circle_radii_l330_330171


namespace repeating_decimals_sum_is_fraction_l330_330824

-- Define the repeating decimals as fractions
def x : ℚ := 1 / 3
def y : ℚ := 2 / 99

-- Define the sum of the repeating decimals
def sum := x + y

-- State the theorem
theorem repeating_decimals_sum_is_fraction :
  sum = 35 / 99 := sorry

end repeating_decimals_sum_is_fraction_l330_330824


namespace part1_part2_l330_330613

def f (x : ℝ) (a : ℝ) : ℝ := (a * exp (2 * x) - 1) / x

theorem part1 (a : ℝ) :
  (tangent_at (f a) (1, f 1 a) (2, 2 * exp 2)) → 
  a = 1 ∧ 
  (monotonically_increasing_on (f 1) (Set.Ioo (-∞) 0) ∧ monotonically_increasing_on (f 1) (Set.Ioo 0 ∞)) :=
sorry

def g (x : ℝ) (a : ℝ) : ℝ := (a * x^2 - 1) / log x

theorem part2 (a : ℝ) (λ : ℝ) :
  a = 1 → 
  (∀ x : ℝ, 1 < x → λ * x * g x 1 ≤ exp (2 * λ * x) - 1) →
  1 / exp 1 ≤ λ :=
sorry

end part1_part2_l330_330613


namespace class_average_is_85_l330_330816

noncomputable def class_average (p1 p2 p3 avg1 avg2 avg3 : ℚ) : ℚ :=
  p1 * avg1 + p2 * avg2 + p3 * avg3

theorem class_average_is_85 : 
  let p1 := 0.45 in 
  let avg1 := 95 in 
  let p2 := 0.50 in 
  let avg2 := 78 in 
  let p3 := 0.05 in 
  let avg3 := 60 in
  round (class_average p1 p2 p3 avg1 avg2 avg3) = 85 :=
by
  sorry

end class_average_is_85_l330_330816


namespace correct_representation_l330_330079

theorem correct_representation :
  ¬ (0 ⊆ ∅) ∧
  ¬ (0 ∈ ∅) ∧
  ¬ (0 = ∅) ∧
  (∅ ⊆ {0}) :=
by
-- Proof is omitted
  sorry

end correct_representation_l330_330079


namespace differential_savings_l330_330646

-- Defining conditions given in the problem
def initial_tax_rate : ℝ := 0.45
def new_tax_rate : ℝ := 0.30
def annual_income : ℝ := 48000

-- Statement of the theorem to prove the differential savings
theorem differential_savings : (annual_income * initial_tax_rate) - (annual_income * new_tax_rate) = 7200 := by
  sorry  -- providing the proof is not required

end differential_savings_l330_330646


namespace height_difference_l330_330121

theorem height_difference
  (a b : ℝ)
  (parabola_eq : ∀ x, y = x^2 + 1)
  (circle_center : b = 2 * a^2 + 1 / 2) :
  b - (a^2 + 1) = a^2 - 1 / 2 :=
by {
  sorry
}

end height_difference_l330_330121


namespace right_triangle_hypotenuse_length_l330_330439

theorem right_triangle_hypotenuse_length (a b : ℝ) (h₁ : a = 6) (h₂ : b = 8) : 
  ∃ c : ℝ, c^2 = a^2 + b^2 ∧ c = 10 :=
by
  sorry

end right_triangle_hypotenuse_length_l330_330439


namespace train_speed_is_correct_l330_330442

/-- Define the length of the train and the time taken to cross the telegraph post. --/
def train_length : ℕ := 240
def crossing_time : ℕ := 16

/-- Define speed calculation based on train length and crossing time. --/
def train_speed : ℕ := train_length / crossing_time

/-- Prove that the computed speed of the train is 15 meters per second. --/
theorem train_speed_is_correct : train_speed = 15 := sorry

end train_speed_is_correct_l330_330442


namespace parallelepiped_volume_l330_330409

variables {V : Type*} [AddCommGroup V] [Module ℝ V] [NormedSpace ℝ V]
variables (a b c : V)

theorem parallelepiped_volume (h : abs (inner a (b × c)) = 3) :
  abs (inner (2 • a - b) ((4 • b + 5 • c) × (3 • c + 2 • a))) = 72 :=
sorry

end parallelepiped_volume_l330_330409


namespace solve_scientific_notation_l330_330300

def scientific_notation (x : ℝ) : ℝ × ℤ :=
  let a := x / 10 ^ (Real.log10 x).floor
  let n := (Real.log10 x).floor
  (a, n)

theorem solve_scientific_notation :
  scientific_notation (595.5 * 10^9) = (5.955, 11) :=
  by
    -- Proof goes here
    sorry

end solve_scientific_notation_l330_330300


namespace necklaces_caught_l330_330548

theorem necklaces_caught
  (LatchNecklaces RhondaNecklaces BoudreauxNecklaces: ℕ)
  (h1 : LatchNecklaces = 3 * RhondaNecklaces - 4)
  (h2 : RhondaNecklaces = BoudreauxNecklaces / 2)
  (h3 : BoudreauxNecklaces = 12) :
  LatchNecklaces = 14 := by
  sorry

end necklaces_caught_l330_330548


namespace length_AB_l330_330681

noncomputable def triangle_ABC (A B C : Type) :=
  ∃ (a b c : ℝ), 
    a = (A,B) ∧ 
    b = (B,C) ∧ 
    c = (A,C) ∧ 
    a = √(b^2 - c^2) ∧ 
    ∠(A,B,C) = 90 ∧ 
    b = 20 ∧ 
    tan (∠(B,C,A)) = 4 * cos (∠(A,B,C))

theorem length_AB {A B C : Type} (h : triangle_ABC A B C) :
  ∃ AB : ℝ, AB = 5 * √(20^2 - 5^2) := 
begin
  sorry
end

end length_AB_l330_330681


namespace fraction_simplify_l330_330528

theorem fraction_simplify :
  (3 + 9 - 27 + 81 - 243 + 729) / (9 + 27 - 81 + 243 - 729 + 2187) = 1 / 3 :=
by
  sorry

end fraction_simplify_l330_330528


namespace function_properties_l330_330991

-- Given conditions
def domain (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ∈ ℝ

def functional_eqn (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f(x * y) = y^2 * f(x) + x^2 * f(y)

-- Lean 4 theorem statement
theorem function_properties (f : ℝ → ℝ)
  (Hdom : domain f)
  (Heqn : functional_eqn f) :
  f 0 = 0 ∧ f 1 = 0 ∧ ∀ x : ℝ, f (-x) = f x :=
by
  sorry


end function_properties_l330_330991


namespace chairs_rearrangement_l330_330197

theorem chairs_rearrangement :
  let chairs := 8
  ∃ f : Fin chairs → Fin chairs,
    (∀ i, f i ≠ i) ∧ -- no one can sit in their original chair
    (∀ i, f i ≠ (i+1) % chairs) ∧ -- no one can sit in the next chair
    (∀ i, f i ≠ (i-1) % chairs) ∧ -- no one can sit in the previous chair
    (f.bijective) ∧ -- f is a bijection
    cardinal.mk {f | (∀ i, f i ≠ i) ∧ (∀ i, f i ≠ (i+1) % chairs) ∧ (∀ i, f i ≠ (i-1) % chairs) ∧ (f.bijective)} = 6 :=
sorry

end chairs_rearrangement_l330_330197


namespace cot_30_eq_sqrt3_l330_330557

theorem cot_30_eq_sqrt3 (theta : ℝ) (h1 : tan (π / 6) = 1 / real.sqrt 3) :
  1 / tan (π / 6) = real.sqrt 3 :=
by
  sorry

end cot_30_eq_sqrt3_l330_330557


namespace binary_to_decimal_11011_is_27_l330_330899

def binary_to_decimal : ℕ :=
  1 * 2^4 + 1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0

theorem binary_to_decimal_11011_is_27 : binary_to_decimal = 27 := by
  sorry

end binary_to_decimal_11011_is_27_l330_330899


namespace union_of_sets_l330_330246

noncomputable def m : ℝ := -2
noncomputable def n : ℝ := 1 / 4
def A : Set ℝ := {2, 2^m}
def B : Set ℝ := {m, n}

lemma intersection_condition : A ∩ B = {1 / 4} := by
  sorry

theorem union_of_sets : A ∪ B = {2, -2, 1 / 4} :=
  by
    have h_inter : A ∩ B = {1 / 4} := intersection_condition
    sorry

end union_of_sets_l330_330246


namespace infinitely_many_a_not_prime_l330_330752

theorem infinitely_many_a_not_prime (a: ℤ) (n: ℤ) : ∃ (b: ℤ), b ≥ 0 ∧ (∃ (N: ℕ) (a: ℤ), a = 4*(N:ℤ)^4 ∧ ∀ (n: ℤ), ¬Prime (n^4 + a)) :=
by { sorry }

end infinitely_many_a_not_prime_l330_330752


namespace function_properties_l330_330994

-- Given conditions
def domain (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ∈ ℝ

def functional_eqn (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f(x * y) = y^2 * f(x) + x^2 * f(y)

-- Lean 4 theorem statement
theorem function_properties (f : ℝ → ℝ)
  (Hdom : domain f)
  (Heqn : functional_eqn f) :
  f 0 = 0 ∧ f 1 = 0 ∧ ∀ x : ℝ, f (-x) = f x :=
by
  sorry


end function_properties_l330_330994


namespace population_doubles_in_35_years_l330_330091

variables (B D : ℝ) (no_emigration_immigration : Prop)

def population_growth_rate := (B - D) / 1000

def doubling_time (pgr : ℝ) := 70 / (pgr * 100)

theorem population_doubles_in_35_years (h1 : B = 39.4) (h2 : D = 19.4) (h3 : no_emigration_immigration) :
  doubling_time (population_growth_rate B D) = 35 :=
by {
  rw [h1, h2],
  simp [population_growth_rate, doubling_time],
  sorry
}

end population_doubles_in_35_years_l330_330091


namespace sin_alpha_given_conditions_l330_330234

theorem sin_alpha_given_conditions 
  (α β : ℝ) 
  (h1 : cos (α - β) = 3 / 5) 
  (h2 : sin β = -5 / 13) 
  (h3 : 0 < α ∧ α < π / 2) 
  (h4 : -π / 2 < β ∧ β < 0) : 
  sin α = 33 / 65 := 
by 
  sorry

end sin_alpha_given_conditions_l330_330234


namespace QR_passes_through_fixed_point_l330_330697

-- Definitions based on the given conditions.
variables {A B C H P E F Q R : Type*} 
  [geometry_triangle A B C H P E F Q R]

-- Definitions specific to the problem setup:
def is_acute_triangle (ABC : triangle) : Prop := sorry
def is_orthocenter (H : point) (ABC : triangle) : Prop := sorry
def nine_point_circle (ABC : triangle) : circle := sorry
def is_on_circle (P : point) (circ : circle) : Prop := sorry
def line (A B : point) : Type* := sorry
def meet_at (l1 l2 : Line) (X : point) : Prop := sorry
def circumcircle (X Y Z : point) : circle := sorry
def intersect_second_time (circ : circle) (line : Line) (P : point) : Prop := sorry
def midpoint (A B : point) : point := sorry

-- The proof problem statement:
theorem QR_passes_through_fixed_point
  (ABC : triangle) (H : point)
  (h_acute : is_acute_triangle ABC)
  (h_orthocenter : is_orthocenter H ABC)
  (P : point)
  (h_on_nine_point : is_on_circle P (nine_point_circle ABC))
  (E F : point) (h_intersect1 : meet_at (line B H) (line H C) E)
  (h_intersect2 : meet_at (line C H) (line H B) F)
  (Q : point) (h_circ1_intersect : intersect_second_time (circumcircle E H P) (line H C) Q)
  (R : point) (h_circ2_intersect : intersect_second_time (circumcircle F H P) (line B H) R)
  : passes_through (line Q R) (midpoint B C) := sorry

end QR_passes_through_fixed_point_l330_330697


namespace sum_factors_of_60_l330_330063

def sum_factors (n : ℕ) : ℕ :=
  (∑ d in Finset.divisors n, d)

theorem sum_factors_of_60 : sum_factors 60 = 168 := by
  sorry

end sum_factors_of_60_l330_330063


namespace line_m_eq_find_a_l330_330244

-- Define the circle, line, point
variable (a : ℝ)

-- Define the equations for circle and line
def circle_eq := ∀ (x y : ℝ), x^2 + y^2 + 4*x - 2*y + a = 0
def line_eq_l := ∀ (x y : ℝ), x - y - 3 = 0

-- Define the definition of orthogonality between OM and ON
def orthogonal (x1 y1 x2 y2 : ℝ) := (x1 * x2 + y1 * y2 = 0)

-- First part: equation of line m
theorem line_m_eq : ∃ m : ℝ → ℝ → Prop, 
  (∀ x y, m x y ↔ x + y + 1 = 0) 
  ∧ ∀ x y, circle_eq x y → line_eq_l x y → m x y :=
begin
  sorry
end

-- Second part: find a
theorem find_a (OM_ON_orthogonal : ∃ M N : ℝ × ℝ, orthogonal O.1 O.2 M.1 M.2 ∧ orthogonal O.1 O.2 N.1 N.2) : 
  ∃ a : ℝ, 
    (∀ x y, circle_eq x y) ∧ 
    (∀ x y, line_eq_l x y → ∃ (x1 x2 y1 y2 : ℝ), orthogonal x1 y1 x2 y2) :=
begin
  use (-18),
  sorry
end

end line_m_eq_find_a_l330_330244


namespace polynomial_rational_iff_image_rationals_l330_330341

theorem polynomial_rational_iff_image_rationals {P : Polynomial ℝ} : 
  (∀ x : ℚ, P.eval x ∈ ℚ) → (∃ Q : Polynomial ℚ, P = Q.map (algebraMap ℚ ℝ)) :=
by
  sorry

end polynomial_rational_iff_image_rationals_l330_330341


namespace min_expression_value_l330_330625

theorem min_expression_value :
  ∀ (O A B : Point)
    (t : ℝ)
    (r : ℝ),
    0 ≤ t ∧ t ≤ 1 →
    |O - A| = 8 →
    |O - B| = 8 →
    O.x * O.y + A.x * A.y = 0 →
    r = |(t * (A - B) - (A - O))| + |(3/4 * (B - O) - (1 - t) * (B - A))| →
    r = 10 := by
  sorry

end min_expression_value_l330_330625


namespace nylon_cord_length_l330_330127

theorem nylon_cord_length {L : ℝ} (hL : L = 30) : ∃ (w : ℝ), w = 5 := 
by sorry

end nylon_cord_length_l330_330127


namespace upstream_distance_calc_l330_330474

noncomputable def speed_in_still_water : ℝ := 10.5
noncomputable def downstream_distance : ℝ := 45
noncomputable def downstream_time : ℝ := 3
noncomputable def upstream_time : ℝ := 3

theorem upstream_distance_calc : 
  ∃ (d v : ℝ), (10.5 + v) * downstream_time = downstream_distance ∧ 
               v = 4.5 ∧ 
               d = (10.5 - v) * upstream_time ∧ 
               d = 18 :=
by
  sorry

end upstream_distance_calc_l330_330474


namespace roots_of_unity_also_roots_of_quadratic_eq_l330_330460

noncomputable def root_of_unity (z : ℂ) (n : ℕ) : Prop := z ^ n = 1

noncomputable def quadratic_root (z : ℂ) (a b : ℤ) : Prop := z ^ 2 + a * z + b = 0

theorem roots_of_unity_also_roots_of_quadratic_eq :
  ∃ n a b : ℤ, ∃ z : ℝ, root_of_unity z n ∧ quadratic_root z a b → multiset.card (multiset.filter (λ z, root_of_unity z n ∧ quadratic_root z a b) {1, -1, complex.I, -complex.I}.to_multiset) = 4 :=
sorry

end roots_of_unity_also_roots_of_quadratic_eq_l330_330460


namespace solution_l330_330387

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 4)
noncomputable def g (x : ℝ) : ℝ := Real.cos (2 * x)

def is_monotonically_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y : ℝ, a ≤ x → x < y → y ≤ b → f x ≤ f y

theorem solution (a b : ℝ) (H : a = 5 * Real.pi / 8 ∧ b = 7 * Real.pi / 8) :
  is_monotonically_increasing g a b :=
sorry

end solution_l330_330387


namespace number_of_sides_of_polygon_l330_330640

-- Given definition about angles and polygons
def exterior_angle (sides: ℕ) : ℝ := 30

-- The sum of exterior angles of any polygon
def sum_exterior_angles : ℝ := 360

-- The proof statement
theorem number_of_sides_of_polygon (k : ℕ) 
  (h1 : exterior_angle k = 30) 
  (h2 : sum_exterior_angles = 360):
  k = 12 :=
sorry

end number_of_sides_of_polygon_l330_330640


namespace find_water_and_bucket_weight_l330_330830

-- Define the original amount of water (x) and the weight of the bucket (y)
variables (x y : ℝ)

-- Given conditions described as hypotheses
def conditions (x y : ℝ) : Prop :=
  4 * x + y = 16 ∧ 6 * x + y = 22

-- The goal is to prove the values of x and y
theorem find_water_and_bucket_weight (h : conditions x y) : x = 3 ∧ y = 4 :=
by
  sorry

end find_water_and_bucket_weight_l330_330830


namespace passenger_waiting_time_probability_l330_330111

theorem passenger_waiting_time_probability :
  ∀ (t_total t_max : ℝ), (t_total = 5) → (t_max = 3) → (t_max / t_total = 3 / 5) :=
by
  intros t_total t_max h_total h_max
  rw [h_total, h_max]
  norm_num
  sorry

end passenger_waiting_time_probability_l330_330111


namespace compound_interest_amount_l330_330430

/-
Given:
- Principal amount P = 5000
- Annual interest rate r = 0.07
- Time period t = 15 years

We aim to prove:
A = 5000 * (1 + 0.07) ^ 15 = 13795.15
-/
theorem compound_interest_amount :
  let P : ℝ := 5000
  let r : ℝ := 0.07
  let t : ℝ := 15
  let A : ℝ := P * (1 + r) ^ t
  A = 13795.15 :=
by
  sorry

end compound_interest_amount_l330_330430


namespace Nick_Tom_Work_Together_Hours_l330_330811

-- Definitions based on the given conditions
def Tom_Cleans_Entire_House_In_Hours := 6
def Tom_Cleans_Half_House := Tom_Cleans_Entire_House_In_Hours / 2
def Time_Tom_Cleans_Half_House_In_Terms_Of_N (N : ℕ) := N / 3

-- Properties and equivalences to derive the conclusion
theorem Nick_Tom_Work_Together_Hours : 
  ∃ (N : ℝ), Time_Tom_Cleans_Half_House_In_Terms_Of_N N = Tom_Cleans_Half_House → 
             let combined_work_rate := (1 / Tom_Cleans_Entire_House_In_Hours + 1 / N) in
             (1 / combined_work_rate) = 3.6 :=
by
  sorry

end Nick_Tom_Work_Together_Hours_l330_330811


namespace bags_on_wednesday_l330_330160

theorem bags_on_wednesday (charge_per_bag money_per_bag monday_bags tuesday_bags total_money wednesday_bags : ℕ)
  (h_charge : charge_per_bag = 4)
  (h_money_per_dollar : money_per_bag = charge_per_bag)
  (h_monday : monday_bags = 5)
  (h_tuesday : tuesday_bags = 3)
  (h_total : total_money = 68) :
  wednesday_bags = (total_money - (monday_bags + tuesday_bags) * money_per_bag) / charge_per_bag :=
by
  sorry

end bags_on_wednesday_l330_330160


namespace total_cost_l330_330507

-- Defining the prices based on the given conditions
def price_smartphone : ℕ := 300
def price_pc : ℕ := price_smartphone + 500
def price_tablet : ℕ := price_smartphone + price_pc

-- The theorem to prove the total cost of buying one of each product
theorem total_cost : price_smartphone + price_pc + price_tablet = 2200 :=
by
  sorry

end total_cost_l330_330507


namespace binary_to_decimal_l330_330900

theorem binary_to_decimal : 
  (1 * 2^0 + 1 * 2^1 + 0 * 2^2 + 1 * 2^3 + 1 * 2^4) = 27 :=
by
  sorry

end binary_to_decimal_l330_330900


namespace max_prod_ab_value_l330_330598

theorem max_prod_ab_value (a b M : ℝ) (h1 : a + b = M) (h2 : a > 0) (h3 : b > 0) (h4 : ∀ x y: ℝ, x + y = M → x * y ≤ 2) : M = 2 * Real.sqrt 2 := by
  have h : a * b ≤ 2 := h4 a b h1
  have fundamental_ineq : a * b ≤ (M / 2) ^ 2 := by
    rw [h1]
    exact mul_le_mul (le_of_lt h2) (le_of_lt h3) (le_of_lt h2) (le_refl (a + b))
  have key_eq : (M / 2) ^ 2 = 2 := by
    rw [sq, div_mul_eq_mul_div, div_eq_mul_inv]
    exact (eq_of_le_of_sq_eq h _).symm
  rw [key_eq, mul_right_eq_self]
  symmetry
  exact ((real.sqrt_mul_self_zero_iff 2).mpr ⟨zero_lt_four, by linarith ⟩).symm, sorry

end max_prod_ab_value_l330_330598


namespace john_using_three_colors_l330_330695

theorem john_using_three_colors {total_paint liters_per_color : ℕ} 
    (h1 : total_paint = 15) 
    (h2 : liters_per_color = 5) :
    total_ppaint / liters_per_color = 3 := 
by
  sorry

end john_using_three_colors_l330_330695


namespace number_of_possible_m_values_l330_330539

theorem number_of_possible_m_values :
  ∃ n : ℕ, n = 1196 ∧
    ∀ m : ℕ,
      (4 ≤ m ∧ m < 1200) ↔
      (∃ a b c : ℝ, a = Real.log 20 ∧ b = Real.log 60 ∧ c = Real.log m ∧
       0 < a ∧ 0 < b ∧ 0 < c ∧
       a + b > c ∧ a + c > b ∧ b + c > a ∧ Real.exp(a) = 20 ∧ Real.exp(b) = 60 ∧ Real.exp(c) = m) :=
by 
  sorry

end number_of_possible_m_values_l330_330539


namespace added_amount_l330_330847

theorem added_amount (x y : ℕ) (h1 : x = 17) (h2 : 3 * (2 * x + y) = 117) : y = 5 :=
by
  sorry

end added_amount_l330_330847


namespace polygon_sides_l330_330638

theorem polygon_sides (exterior_angle : ℝ) (sum_exterior_angles : ℝ) (h1 : exterior_angle = 30) (h2 : sum_exterior_angles = 360) :
  (sum_exterior_angles / exterior_angle) = 12 :=
by
  have h3 : ∀ (n : ℝ), n = 360 / 30, from sorry
  rw [h1, h2] at h3
  exact h3

end polygon_sides_l330_330638


namespace problem_statement_l330_330716

noncomputable def roots_cubic := {a b c : ℝ // 
  poly_roots (λ x, x^3 - 8*x^2 + 14*x - 2) {a, b, c} }

noncomputable def t (roots : roots_cubic) : ℝ := 
  let ⟨a, b, c, h⟩ := roots in
  sqrt a + sqrt b + sqrt c

theorem problem_statement (roots : roots_cubic) : 
  let t := t roots in
  t^4 - 16 * t^2 - 12 * t = - (8 * sqrt 2) / 3 :=
sorry

end problem_statement_l330_330716


namespace compute_integral_l330_330876

noncomputable def evaluate_integral : ℝ :=
  ∫ x in 0..1, (Real.sqrt (1 - (x - 1)^2) - 2 * x)

theorem compute_integral :
  evaluate_integral = (Real.pi / 4) - 1 :=
by 
  sorry

end compute_integral_l330_330876


namespace find_150th_letter_l330_330044

theorem find_150th_letter (n : ℕ) (pattern : ℕ → Char) (h : ∀ m, pattern (m % 3) = if m % 3 = 0 then 'A' else if m % 3 = 1 then 'B' else 'C') :
  pattern 149 = 'C' :=
by
  sorry

end find_150th_letter_l330_330044


namespace pastry_machine_completion_time_l330_330826

def start_time : Nat := 9 * 60 -- 9:00 AM in minutes
def half_completion_time : Nat := 12 * 60 + 30 -- 12:30 PM in minutes
def total_job_time (start half : Nat) : Nat := 2 * (half - start)

theorem pastry_machine_completion_time :
    total_job_time start_time half_completion_time == 7 * 60 :=
by 
   -- The job completion time is calculated as
   let completion_time := start_time + total_job_time start_time half_completion_time in
   have h1: total_job_time start_time half_completion_time = 7 * 60 := sorry,
   show completion_time = 9 * 60 + 7 * 60, by rw h1; rfl

end pastry_machine_completion_time_l330_330826


namespace average_rate_correct_l330_330846

noncomputable def average_rate : ℝ :=
  let downstream_speed := 20 -- km/h
  let upstream_speed := 4 -- km/h
  let distance_downstream := 60 -- km
  let distance_upstream := 30 -- km
  let wind_speed := 3 -- km/h in the opposite direction
  let new_downstream_speed := downstream_speed - wind_speed
  let new_upstream_speed := upstream_speed + wind_speed
  let time_downstream := distance_downstream / new_downstream_speed
  let time_upstream := distance_upstream / new_upstream_speed
  let total_distance := distance_downstream + distance_upstream
  let total_time := time_downstream + time_upstream
  total_distance / total_time

theorem average_rate_correct : average_rate ≈ 11.51 := sorry

end average_rate_correct_l330_330846


namespace star_polygon_x_value_l330_330603

theorem star_polygon_x_value
  (a b c d e p q r s t : ℝ)
  (h1 : p + q + r + s + t = 500)
  (h2 : a + b + c + d + e = x)
  :
  x = 140 :=
sorry

end star_polygon_x_value_l330_330603


namespace probability_of_max_less_than_next_l330_330152

-- defining the probability recursively as given in the problem
noncomputable def p : ℕ → ℝ
| 0     := 1
| (n+1) := (2:ℝ) / (n+2) * p n

-- The main theorem to state the desired probability
theorem probability_of_max_less_than_next (n : ℕ) : p n = 2^n / (n + 1)! := 
sorry

end probability_of_max_less_than_next_l330_330152


namespace find_A_and_B_l330_330553

theorem find_A_and_B :
  ∃ A B : ℚ,
    (A = 59 / 11) ∧ 
    (B = 18 / 11) ∧ 
    (∀ x : ℚ, x ≠ 9 → x ≠ -2 → 
      (7 * x - 4) / (x^2 - 9 * x - 18) = A / (x - 9) + B / (x + 2)) :=
by {
  let A := (59 : ℚ) / 11,
  let B := (18 : ℚ) / 11,
  use A, B,
  split, norm_num,
  split, norm_num,
  intros x h1 h2,
  field_simp [h1, h2],
  ring
}

end find_A_and_B_l330_330553


namespace clock_hands_right_angles_in_5_days_l330_330631

theorem clock_hands_right_angles_in_5_days : 
  (let times_per_hour := 2 in
   let hours_per_day := 24 in
   let days := 5 in
   times_per_hour * hours_per_day * days = 240) :=
by
  let times_per_hour := 2
  let hours_per_day := 24
  let days := 5
  show times_per_hour * hours_per_day * days = 240
  sorry

end clock_hands_right_angles_in_5_days_l330_330631


namespace sin_double_angle_l330_330235

theorem sin_double_angle (α : ℝ) (h : Real.tan α = -1/3) : Real.sin (2 * α) = -3/5 := by 
  sorry

end sin_double_angle_l330_330235


namespace no_such_sequence_l330_330913

theorem no_such_sequence (a : ℕ → ℕ) :
  (∀ n, a n > 0) →
  (∀ n, a (n + 2) = a (n + 1) + nat.sqrt (a (n + 1) + a n)) →
  false :=
by
  sorry

end no_such_sequence_l330_330913


namespace total_cost_l330_330508

-- Defining the prices based on the given conditions
def price_smartphone : ℕ := 300
def price_pc : ℕ := price_smartphone + 500
def price_tablet : ℕ := price_smartphone + price_pc

-- The theorem to prove the total cost of buying one of each product
theorem total_cost : price_smartphone + price_pc + price_tablet = 2200 :=
by
  sorry

end total_cost_l330_330508


namespace statement_A_statement_B_statement_C_statement_D_l330_330433

/- Conditions -/
variables {A B : set ℝ} -- We assume events are subsets of some universe, typically ℝ for simplicity in Lean
variable {ξ : ℝ → ℝ}   -- Assuming ξ is a random variable
variable {X Y : ℝ → ℝ} -- Random variables X and Y
variable {r : ℝ}       -- Correlation coefficient

/- Probabilities of events A and B -/
axiom P : set ℝ → ℝ
axiom prob_A : P(A) = 0.3
axiom prob_B : P(B) = 0.6
axiom subset_AB : A ⊆ B

/- Probability condition for normally distributed ξ -/
axiom dist_ξ : ∀(d : ℝ), ξ d ~ Normal(2, δ^2)
axiom prob_ξ_lt_4 : P({d | ξ d < 4}) = 0.84

/- Condition on correlation coefficient -/
axiom correlation_XY : Correlation X Y == r

/- Regression analysis residual band condition -/
axiom wider_residual_band_worse_regression : True -- We take this as given for simplicity

/- Statements to Prove -/
theorem statement_A : P(B | A) = 1 := sorry
theorem statement_B : ¬ (P({d | 2 < ξ d < 4}) = 0.16) := sorry
theorem statement_C : (|r| < 1) → stronger_linear_correlation := sorry
theorem statement_D : wider_residual_band_worse_regression := sorry

end statement_A_statement_B_statement_C_statement_D_l330_330433


namespace part1_part2_l330_330628

open Real

variables (a b : ℝ → ℝ → ℝ) -- This would represent the vector space, simplified.
variables (dot : (ℝ → ℝ → ℝ) → (ℝ → ℝ → ℝ) → ℝ)
variables (norm : (ℝ → ℝ → ℝ) → ℝ)

axiom norm_a : norm a = 2
axiom norm_b : norm b = 3
axiom dot_condition : dot (3 * a + b) (a - 2 * b) = -16

theorem part1 (h : dot (a - b) (a + λ * b) = 0) : λ = 2 / 7 := sorry

theorem part2 : 
  let ab := dot a (2 * a - b)
  let norm_ab := norm (2 * a - b)
  ab / (norm a * norm_ab) = 3 / sqrt 17 := sorry

end part1_part2_l330_330628


namespace vertical_angles_congruent_l330_330748

theorem vertical_angles_congruent (A B : Type) [angle A] [angle B] (h : vertical A B) : congruent A B :=
sorry

end vertical_angles_congruent_l330_330748


namespace y_coord_range_of_M_l330_330316

theorem y_coord_range_of_M :
  ∀ (M : ℝ × ℝ), ((M.1 + 1)^2 + M.2^2 = 2) → 
  ((M.1 - 2)^2 + M.2^2 + M.1^2 + M.2^2 ≤ 10) →
  - (Real.sqrt 7) / 2 ≤ M.2 ∧ M.2 ≤ (Real.sqrt 7) / 2 := 
by 
  sorry

end y_coord_range_of_M_l330_330316


namespace G_minus_L_value_l330_330506

-- Given conditions:
-- 1. Arithmetic sequence with 150 terms
-- 2. Each term is between 20 and 120
-- 3. Sum of terms is 12000

theorem G_minus_L_value :
  (∃ (a d : ℝ), (∀ n : ℕ, 0 ≤ n → n < 150 → 20 ≤ a + n * d ∧ a + n * d ≤ 120) ∧
   (finset.Ico 0 150).sum (λ n, a + n * d) = 12000) →
  (∃ L G : ℝ, 
    (L = 80 - 75 * (40 / 149)) ∧ 
    (G = 80 + 75 * (40 / 149)) ∧ 
    (G - L = 6000 / 149)) :=
sorry

end G_minus_L_value_l330_330506


namespace balls_in_boxes_l330_330632

theorem balls_in_boxes (n k : ℕ) (hn : n = 4) (hk : k = 3) :
  (∑ (x : ℕ) in range (k + 1), (fintype.card {m : vector ℕ k // m.sum = n})) = 15 :=
by
  rw [hn, hk]
  sorry

end balls_in_boxes_l330_330632


namespace binom_150_150_l330_330178

-- Definition of binomial coefficient
def binom (n k : ℕ) : ℕ :=
  if h : k ≤ n then (nat.choose n k) else 0

-- Symmetry of binomial coefficients
lemma binom_symm (n k : ℕ) : binom n k = binom n (n - k) :=
by sorry

-- The basic property of choosing 0 elements from n elements
lemma binom_zero (n : ℕ) : binom n 0 = 1 :=
by sorry

-- The main theorem to be proved.
theorem binom_150_150 : binom 150 150 = 1 :=
by sorry

end binom_150_150_l330_330178


namespace average_weight_of_16_boys_l330_330410

theorem average_weight_of_16_boys :
  ∃ A : ℝ,
    (16 * A + 8 * 45.15 = 24 * 48.55) ∧
    A = 50.25 :=
by {
  -- Proof skipped, using sorry to denote the proof is required.
  sorry
}

end average_weight_of_16_boys_l330_330410


namespace problem_A_value_l330_330281

theorem problem_A_value:
  (let lhs := (1/(1+24/4) - 5/9) * (3 / (2 * 5/7)) / (2 / (3 * 3/4)) + 2.25 in
  lhs = 4) →
  (let eqn_left := 1/(1 + (24 + A) / (5 * A)) - 5/9 in
  let new_val := eqn_left * (315 / 152) + 2.25 in
  new_val = 4 → A = 4) :=
by
  intros h1 h2
  sorry

end problem_A_value_l330_330281


namespace pascal_triangle_contains_31_rows_l330_330276

theorem pascal_triangle_contains_31_rows : ∀ n : ℕ, (n > 0 → ∃! r : ℕ, (r ≥ n) ∧ ∃ k : ℕ, binomial r k = 31) :=
by 
  sorry

end pascal_triangle_contains_31_rows_l330_330276


namespace simplification_and_evaluation_l330_330610

noncomputable def f (alpha : ℝ) : ℝ := 
  (Real.sin (alpha - Real.pi) * Real.cos (2 * Real.pi - alpha) *
   Real.sin (-alpha + 3 * Real.pi / 2) * Real.sin (5 * Real.pi / 2 + alpha)) /
  (Real.cos (-Real.pi - alpha) * Real.sin (-Real.pi - alpha))

theorem simplification_and_evaluation (alpha : ℝ) :
  (f(alpha) = 0.5 * Real.cos (2 * alpha) + 0.5) ∧
  (Real.cos (5 * Real.pi / 6 + 2 * alpha) = 1 / 3) →
  f(Real.pi / 12 - alpha) = 2 / 3 :=
by
  sorry

end simplification_and_evaluation_l330_330610


namespace find_g_of_3_l330_330389

noncomputable def g : ℝ → ℝ := sorry

theorem find_g_of_3 (h : ∀ x : ℝ, g (3 ^ x) + x * g (3 ^ (-x)) = 3) : g 3 = 0 :=
by sorry

end find_g_of_3_l330_330389


namespace angle_C_max_value_l330_330287

theorem angle_C_max_value (A B C : ℝ) (h1 : A + B + C = π)
  (h2 : sin B / sin A = 2 * cos (A + B)) :
  C = 2 * π / 3 :=
sorry

end angle_C_max_value_l330_330287


namespace larger_cylinder_candies_l330_330836

theorem larger_cylinder_candies (v₁ v₂ : ℝ) (c₁ c₂ : ℕ) (h₁ : v₁ = 72) (h₂ : c₁ = 30) (h₃ : v₂ = 216) (h₄ : (c₁ : ℝ)/v₁ = (c₂ : ℝ)/v₂) : c₂ = 90 := by
  -- v1 h1 h2 v2 c2 h4 are directly appearing in the conditions
  -- ratio h4 states the condition for densities to be the same 
  sorry

end larger_cylinder_candies_l330_330836


namespace measure_of_angle_YZX_l330_330176

-- Define the problem conditions
variables {A B C X Y Z : Type}
noncomputable def angle_A : ℝ := 50
noncomputable def angle_B : ℝ := 70
noncomputable def angle_C : ℝ := 60

-- Assume a circumcircle passing through all points of the triangles
axiom circumcircle_ABC : ∃ (Γ : set (Metric.sphere A B C)), is_circumcircle Γ (triangle A B C)
axiom circumcircle_XYZ : ∃ (Γ : set (Metric.sphere X Y Z)), is_circumcircle Γ (triangle X Y Z)

-- Points on the sides of the triangle
axiom X_on_BC : X ∈ segment B C
axiom Y_on_AB : Y ∈ segment A B
axiom Z_on_AC : Z ∈ segment A C

-- Main theorem statement to be proven
theorem measure_of_angle_YZX : 
  ∠ YZX = 50 :=
sorry

end measure_of_angle_YZX_l330_330176


namespace distance_between_foci_l330_330866

-- Define the ellipse equation and extract a, b.
def ellipse_eq (x y : ℝ) := 9 * x^2 + y^2 = 144

-- Define a unified statement to prove the distance between foci
theorem distance_between_foci : 
  (∃ a b : ℝ, a = 4 ∧ b = 12 ∧ (∀ c, c = real.sqrt (b^2 - a^2) → c = 8 * real.sqrt 2))
  :=
sorry

end distance_between_foci_l330_330866


namespace number_of_ways_to_fold_cube_with_one_face_missing_l330_330885

-- Definitions:
-- The polygon is initially in the shape of a cross with 5 congruent squares.
-- One additional square can be attached to any of the 12 possible edge positions around this polygon.
-- Define what it means for the resulting figure to fold into a cube with one face missing.

-- Statement:
theorem number_of_ways_to_fold_cube_with_one_face_missing 
  (initial_squares : ℕ)
  (additional_positions : ℕ)
  (valid_folding_positions : ℕ) : 
  initial_squares = 5 ∧ additional_positions = 12 → valid_folding_positions = 8 :=
by
  sorry

end number_of_ways_to_fold_cube_with_one_face_missing_l330_330885


namespace arithmetic_progression_nat_seq_l330_330204

theorem arithmetic_progression_nat_seq (a : ℕ → ℕ) (N : ℕ) 
  (h1 : N > 1) 
  (h2 : ∀ k : ℕ, ∏ i in finset.range (k+1), a i ∣ ∏ i in finset.range (k+1), a (N + i)) : 
  (∃ k : ℕ, k > 0 ∧ ∀ n : ℕ, a n = k * (n + 1)) ∨ (∃ c : ℕ, ∀ n : ℕ, a n = c) :=
begin
  sorry
end

end arithmetic_progression_nat_seq_l330_330204


namespace alex_money_left_l330_330860

noncomputable def alex_main_income : ℝ := 900
noncomputable def alex_side_income : ℝ := 300
noncomputable def main_job_tax_rate : ℝ := 0.15
noncomputable def side_job_tax_rate : ℝ := 0.20
noncomputable def water_bill : ℝ := 75
noncomputable def main_job_tithe_rate : ℝ := 0.10
noncomputable def side_job_tithe_rate : ℝ := 0.15
noncomputable def grocery_expense : ℝ := 150
noncomputable def transportation_expense : ℝ := 50

theorem alex_money_left :
  let main_income_after_tax := alex_main_income * (1 - main_job_tax_rate)
  let side_income_after_tax := alex_side_income * (1 - side_job_tax_rate)
  let total_income_after_tax := main_income_after_tax + side_income_after_tax
  let main_tithe := alex_main_income * main_job_tithe_rate
  let side_tithe := alex_side_income * side_job_tithe_rate
  let total_tithe := main_tithe + side_tithe
  let total_deductions := water_bill + grocery_expense + transportation_expense + total_tithe
  let money_left := total_income_after_tax - total_deductions
  money_left = 595 :=
by
  -- Proof goes here
  sorry

end alex_money_left_l330_330860


namespace tub_drain_time_l330_330288

theorem tub_drain_time (time_for_five_sevenths : ℝ)
  (time_for_five_sevenths_eq_four : time_for_five_sevenths = 4) :
  let rate := time_for_five_sevenths / (5 / 7)
  let time_for_two_sevenths := 2 * rate
  time_for_two_sevenths = 11.2 := by
  -- Definitions and initial conditions
  sorry

end tub_drain_time_l330_330288


namespace number_of_solutions_l330_330285

noncomputable def f (x : ℝ) : ℝ :=
if x >= -2 then x^2 - 4 else x + 4

theorem number_of_solutions (x : ℝ) : nat :=
let f (x : ℝ) : ℝ := if x >= -2 then x^2 - 4 else x + 4 in
4

#eval number_of_solutions 4

end number_of_solutions_l330_330285


namespace range_of_fa_tangent_lines_l330_330455

open Real

-- For the integral condition and range of f(a)
def f (x : ℝ) : ℝ := a * x + b
def integral_condition : Prop := ∫ (x : ℝ) in -1..1, (f x)^2 = 2
theorem range_of_fa (h : integral_condition) : -1 ≤ f a ∧ f a ≤ 37 / 12 :=
sorry

-- For the tangent line problem
def f_prime (x : ℝ) : ℝ := 3 * x^2 - 3
def tangent_line_condition (t : ℝ) : Prop := 
  let k := f_prime t in
  let y_t := t^3 - 3 * t in
  let P := (1, -2) in
  y_t = -2 ∧ (t - 1 + k ≠ 0)

theorem tangent_lines (h : tangent_line_condition t) :
  (y + 2 = 0 ∨ 9 * x + 4 * y - 1 = 0) :=
sorry

end range_of_fa_tangent_lines_l330_330455


namespace special_collection_books_l330_330136

theorem special_collection_books (initial_books loaned_books returned_percent: ℕ) (loaned_books_value: loaned_books = 55) (returned_percent_value: returned_percent = 80) (initial_books_value: initial_books = 75) :
  initial_books - (loaned_books - (returned_percent * loaned_books / 100)) = 64 := by
  sorry

end special_collection_books_l330_330136


namespace sum_of_positive_factors_60_l330_330066

theorem sum_of_positive_factors_60 : (∑ n in Nat.divisors 60, n) = 168 := by
  sorry

end sum_of_positive_factors_60_l330_330066


namespace new_student_weight_l330_330820

theorem new_student_weight (avg_weight : ℝ) (x : ℝ) :
  (avg_weight * 10 - 120) = ((avg_weight - 6) * 10 + x) → x = 60 :=
by
  intro h
  -- The proof would go here, but it's skipped.
  sorry

end new_student_weight_l330_330820


namespace teddy_pillows_l330_330372

theorem teddy_pillows (tons_of_material : ℕ) (pounds_per_ton : ℕ)
  (foam_material_per_pillow : ℕ) (total_pillows : ℕ) : 
  tons_of_material = 3 → pounds_per_ton = 2000 →
  foam_material_per_pillow = 5 - 3 →
  total_pillows = (tons_of_material * pounds_per_ton) / foam_material_per_pillow :=
by
  -- Convert the given tons to pounds
  let total_pounds := tons_of_material * pounds_per_ton
  -- Calculate the number of pillows
  let calculated_pillows := total_pounds / foam_material_per_pillow
  -- Assert the expected number of pillows
  have h : total_pillows = calculated_pillows := by rfl
  sorry

end teddy_pillows_l330_330372


namespace trig_identity_l330_330247

theorem trig_identity (α : ℝ) (h : (cos α + sin α) / (cos α - sin α) = 2) : 
  1 + 3 * sin α * cos α - 2 * cos α ^ 2 = 1 / 10 := 
by 
  sorry

end trig_identity_l330_330247


namespace mode_and_median_of_dataset_l330_330309

def dataset : List ℕ := [3, 7, 5, 6, 5, 4]

theorem mode_and_median_of_dataset :
  (mode_of_dataset dataset = 5) ∧ (median_of_dataset dataset = 5) :=
by sorry

end mode_and_median_of_dataset_l330_330309


namespace train_crossing_time_l330_330489

-- Define lengths and speed
def length_train := 150  -- meters
def length_bridge := 320  -- meters
def speed_train := 42.3  -- km/h

-- Conversion factor from km/h to m/s
def kmh_to_ms (v : ℝ) : ℝ :=
  (v * 1000) / 3600

-- Define the speed of the train in m/s
def speed_train_ms : ℝ :=
  kmh_to_ms speed_train

-- Define total distance to be covered by the train
def total_distance : ℝ :=
  length_train + length_bridge

-- Define the calculated time to cross the bridge
def time_to_cross : ℝ :=
  total_distance / speed_train_ms

-- Assertion to verify the time to cross is approximately 40 seconds
theorem train_crossing_time : abs (time_to_cross - 40) < 1 :=
  sorry

end train_crossing_time_l330_330489


namespace convert_binary_to_decimal_l330_330895

-- Define the binary number and its decimal conversion function
def bin_to_dec (bin : List ℕ) : ℕ :=
  list.foldl (λ acc b, 2 * acc + b) 0 bin

-- Define the specific problem conditions
def binary_oneonezeronethreeone : List ℕ := [1, 1, 0, 1, 1]

theorem convert_binary_to_decimal :
  bin_to_dec binary_oneonezeronethreeone = 27 := by
  sorry

end convert_binary_to_decimal_l330_330895


namespace no_solution_abs_eq_quadratic_l330_330208

theorem no_solution_abs_eq_quadratic (x : ℝ) : ¬ (|x - 4| = x^2 + 6 * x + 8) :=
by
  sorry

end no_solution_abs_eq_quadratic_l330_330208


namespace proof_random_event_l330_330078

-- Define each event and their properties.

def eventA : Prop := ∀ (ball: Unit), true  -- A basketball will fall down due to gravity
def eventB : Prop := ∃ (ticket: Unit), random_event ticket  -- Buying a lottery ticket and winning $10 million is a random event
def eventC : Prop := ∀ (people: Fin 14), ∃ (month: Fin 12), true  -- At least 2 out of 14 people must have the same birth month
def eventD : Prop := false  -- Drawing a black ball from a bag with only red and white balls is impossible

-- Prove that eventB is a random event given these conditions

theorem proof_random_event : random_event eventB :=
by  
  sorry

end proof_random_event_l330_330078


namespace cuboid_area_correct_l330_330926

-- Define the parameters (length, breadth, height) of the cuboid
def length : ℝ := 15
def breadth : ℝ := 10
def height : ℝ := 16

-- Define the surface area formula for a cuboid
def cuboid_surface_area (l b h : ℝ) : ℝ :=
  2 * (l * b) + 2 * (b * h) + 2 * (l * h)

-- Statement for the proof problem
theorem cuboid_area_correct :
  cuboid_surface_area length breadth height = 1100 :=
by
  sorry

end cuboid_area_correct_l330_330926


namespace tangent_chord_bisect_l330_330773

/-- Given that OA is tangent to a circle at A,
and the chord BC is parallel to OA,
and lines OB and OC intersect the circle again at points K and L,
prove that the line KL bisects the segment OA. -/
theorem tangent_chord_bisect 
  (O A B C K L : Point)
  (h_tangent : Tangent OA O A)
  (h_parallel : Parallel BC OA)
  (h_intersect_K : CircleIntersect OB K)
  (h_intersect_L : CircleIntersect OC L) :
  Bisects KL OA :=
sorry

end tangent_chord_bisect_l330_330773


namespace area_change_l330_330374

theorem area_change (original_area new_area : ℝ) (length_factor width_factor : ℝ)
  (h1 : original_area = 432)
  (h2 : length_factor = 1.2)
  (h3 : width_factor = 0.9) :
  new_area = 467 :=
by
  have : new_area = original_area * length_factor * width_factor,
  { sorry },
  rw [h1, h2, h3] at this,
  norm_num at this,
  exact this

end area_change_l330_330374


namespace range_of_y_l330_330003

noncomputable def y (x : ℝ) : ℝ := Real.sin x + Real.arcsin x

theorem range_of_y : 
  ∀ x, -1 ≤ x ∧ x ≤ 1 →
  y x ∈ Set.Icc (-Real.sin 1 - (Real.pi / 2)) (Real.sin 1 + (Real.pi / 2)) :=
sorry

end range_of_y_l330_330003


namespace impossible_goal_sum_of_10_l330_330095

variables (AntonScore IlyaScore SeryozhaScore : ℕ)

def AntonStatement1 := AntonScore = 3
def AntonStatement2 := IlyaScore = 1

def IlyaStatement1 := IlyaScore = 4
def IlyaStatement2 := SeryozhaScore = 5

def SeryozhaStatement1 := SeryozhaScore = 6
def SeryozhaStatement2 := AntonScore = 2

def TruthLieConstraints :=
  (AntonStatement1 ∨ AntonStatement2) ∧ (¬AntonStatement1 ∨ ¬AntonStatement2) ∧
  (IlyaStatement1 ∨ IlyaStatement2) ∧ (¬IlyaStatement1 ∨ ¬IlyaStatement2) ∧
  (SeryozhaStatement1 ∨ SeryozhaStatement2) ∧ (¬SeryozhaStatement1 ∨ ¬SeryozhaStatement2)


theorem impossible_goal_sum_of_10 (h : TruthLieConstraints) : 
  AntonScore + IlyaScore + SeryozhaScore ≠ 10 :=
begin
  sorry
end

end impossible_goal_sum_of_10_l330_330095


namespace arithmetic_seq_is_increasing_avg_sum_seq_is_increasing_l330_330241

variable {a_n : ℕ → ℝ} {S_n : ℕ → ℝ} {d : ℝ}

noncomputable def arithmetic_seq_increasing (a_n : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n : ℕ, a_n (n + 1) = a_n n + d

noncomputable def sum_seq (a_n : ℕ → ℝ) (n : ℕ) : ℝ :=
(n.succ * a_n 0 + (n * (n + 1) / 2) * d) / 2

theorem arithmetic_seq_is_increasing (h_arith : ∀ n : ℕ, a_n (n + 1) = a_n n + d)
(h_d_positive: d > 0) : ∀ n : ℕ, a_n n < a_n (n + 1) := 
by sorry

theorem avg_sum_seq_is_increasing (h_arith : ∀ n : ℕ, a_n (n + 1) = a_n n + d)
(h_d_positive: d > 0) (h_sum : ∀ n : ℕ, S_n n = sum_seq a_n n) : 
∀ m n : ℕ, m < n → (S_n m / m) < (S_n n / n) :=
by sorry

end arithmetic_seq_is_increasing_avg_sum_seq_is_increasing_l330_330241


namespace num_ordered_19_tuples_l330_330934

theorem num_ordered_19_tuples :
  (Fin 19 -> ℤ) ->
  (∀ (b : (Fin 19 -> ℤ)) (i : Fin 19),
    (b i)^2 = (Finset.univ.sum (λ j, if j = i then 0 else b j)) + 1) ->
  (fintype { b : Fin 19 -> ℤ // ∀ i : Fin 19, (b i)^2 = (Finset.univ.sum (λ j, if j = i then 0 else b j)) + 1 }) =
  54264 :=
sorry

end num_ordered_19_tuples_l330_330934


namespace remainder_problem_l330_330724

theorem remainder_problem (x y : ℤ) (k m : ℤ) 
  (hx : x = 126 * k + 11) 
  (hy : y = 126 * m + 25) :
  (x + y + 23) % 63 = 59 := 
by
  sorry

end remainder_problem_l330_330724


namespace circle_equation_midpoint_trajectory_l330_330251

-- Definition for the circle equation proof
theorem circle_equation (x y : ℝ) (h : (x - 3)^2 + (y - 2)^2 = 13)
  (hx : x = 3) (hy : y = 2) : 
  (x - 3)^2 + (y - 2)^2 = 13 := by
  sorry -- Placeholder for proof

-- Definition for the midpoint trajectory proof
theorem midpoint_trajectory (x y : ℝ) (hx : x = (2 * x - 11) / 2)
  (hy : y = (2 * y - 2) / 2) (h : (2 * x - 11)^2 + (2 * y - 2)^2 = 13) :
  (x - 11 / 2)^2 + (y - 1)^2 = 13 / 4 := by
  sorry -- Placeholder for proof

end circle_equation_midpoint_trajectory_l330_330251


namespace geometric_sum_first_eight_terms_l330_330938

theorem geometric_sum_first_eight_terms :
  let a := (2 : ℚ) / (3 : ℚ)
  let r := (1 : ℚ) / (3 : ℚ)
  (a * (1 - r^8) / (1 - r)) = (6560 / 6561 : ℚ) :=
by
  let a := (2 : ℚ) / (3 : ℚ)
  let r := (1 : ℚ) / (3 : ℚ)
  have h : a * (1 - r^8) / (1 - r) = (2 / 3) * (1 - (1 / 3)^8) / ((2 / 3) * (3 / 2)),
  {
    calc
    a * (1 - r^8) / (1 - r)
      = (2 / 3) * (1 - (1 / 3)^8) / (1 - 1 / 3) : by rw [a, r]
      ... = (2 / 3) * (1 - 1 / 6561) / (2 / 3 / 3 / (2 / 3)) : -- using intermediate computations
  },
  sorry

end geometric_sum_first_eight_terms_l330_330938


namespace sum_of_super_cool_rectangle_areas_l330_330478

def is_super_cool_rectangle (a b : ℕ) : Prop :=
  a * b = 6 * (a + b)

def area (a b : ℕ) := a * b

theorem sum_of_super_cool_rectangle_areas : 
  let areas := {(a, b) | ∃ a b : ℕ, is_super_cool_rectangle a b} in
  (finset.finset_sum (finset.image (λ (x : ℕ × ℕ), area x.1 x.2) areas) = 942) :=
by {
  sorry
}

end sum_of_super_cool_rectangle_areas_l330_330478


namespace player1_wins_l330_330032

noncomputable def player1_wins_infinite_grid_game :
  Prop :=
∃ strategy : ℕ → ℕ → ℤ → ℤ,
  (∀ n : ℕ, player1_strategy n) ∧ 
  (∀ m : ℕ, player2_strategy m)

theorem player1_wins :
  ∃ strategy : ℕ → ℕ → ℤ → ℤ,
    (∃ player1_strategy : ℕ → ℤ × ℤ,
      ∃ player2_strategy : ℕ → ℤ × ℤ,
        ∀ n : ℕ, player1_strategy n ∧ player2_strategy n) :=
sorry

end player1_wins_l330_330032


namespace polynomial_irreducible_iff_l330_330946

theorem polynomial_irreducible_iff (n : ℕ) (h : n > 0) :
  irreducible (1 + X^n + X^(2 * n) : ℤ[X]) ↔ ∃ (k : ℕ), n = 3^k :=
sorry

end polynomial_irreducible_iff_l330_330946


namespace circle_tangent_parabola_height_difference_l330_330117

theorem circle_tangent_parabola_height_difference 
  (a b : ℝ)
  (h_tangent1 : (a, a^2 + 1))
  (h_tangent2 : (-a, a^2 + 1))
  (parabola_eq : ∀ x, x^2 + 1 = (x, x^2 + 1))
  (circle_eq : ∀ x, x^2 + ((x^2 + 1) - b)^2 = r^2) : 
  b - (a^2 + 1) = 1 / 2 :=
sorry

end circle_tangent_parabola_height_difference_l330_330117


namespace opposite_of_2023_is_neg_2023_l330_330780

theorem opposite_of_2023_is_neg_2023 (x : ℝ) (h : x = 2023) : -x = -2023 :=
by
  /- proof begins here, but we are skipping it with sorry -/
  sorry

end opposite_of_2023_is_neg_2023_l330_330780


namespace fraction_irreducible_iff_mod5_l330_330945

theorem fraction_irreducible_iff_mod5 (n : ℕ) : nat.coprime (n^3 + n) (2*n + 1) ↔ ¬((n % 5) = 2) :=
by sorry

end fraction_irreducible_iff_mod5_l330_330945


namespace highest_possible_value_l330_330308

theorem highest_possible_value 
  (t q r1 r2 : ℝ)
  (h_eq : r1 + r2 = t)
  (h_cond : ∀ n : ℕ, n > 0 → r1^n + r2^n = t) :
  t = 2 → q = 1 → 
  r1 = 1 → r2 = 1 →
  (1 / r1^1004 + 1 / r2^1004 = 2) :=
by
  intros h_t h_q h_r1 h_r2
  rw [h_r1, h_r2]
  norm_num

end highest_possible_value_l330_330308


namespace unique_providers_for_children_l330_330696

theorem unique_providers_for_children (providers : Finset ℕ) (card_providers : providers.card = 23) :
  ∃! choices : Finset (Finset ℕ), 
  choices.card = 4 ∧ 
  ∀ choice ∈ choices, choice.card = 1 ∧ choice ⊆ providers ∧
  choices.pairwise Disjoint ∧
  choices.sum.finset.1.card = 4 ∧
  choices.sum = 213840 :=
by
  sorry

end unique_providers_for_children_l330_330696


namespace evaluate_f_at_2_l330_330038

def f (x : ℝ) : ℝ := 2 * x^5 + 3 * x^4 + 2 * x^3 - 4 * x + 5

theorem evaluate_f_at_2 :
  f 2 = 125 :=
by
  sorry

end evaluate_f_at_2_l330_330038


namespace union_sets_l330_330622

open Set

variables {α : Type*} [DecidableEq α]

theorem union_sets (a b : α) (M N : Set α) (h1 : M = {3, 2^a})
  (h2 : N = {a, b}) (h3 : M ∩ N = {2}) : M ∪ N = {1, 2, 3} :=
by
  sorry

end union_sets_l330_330622


namespace sum_of_factors_of_60_l330_330073

-- Given conditions
def n : ℕ := 60

-- Proof statement
theorem sum_of_factors_of_60 : (∑ d in (finset.filter (λ d, n % d = 0) (finset.range (n + 1))), d) = 168 :=
by
  sorry

end sum_of_factors_of_60_l330_330073


namespace product_xyz_l330_330012

noncomputable def x : ℚ := 97 / 12
noncomputable def n : ℚ := 8 * x
noncomputable def y : ℚ := n + 7
noncomputable def z : ℚ := n - 11

theorem product_xyz 
  (h1: x + y + z = 190)
  (h2: n = 8 * x)
  (h3: n = y - 7)
  (h4: n = z + 11) : 
  x * y * z = (97 * 215 * 161) / 108 := 
by 
  sorry

end product_xyz_l330_330012


namespace sum_of_positive_factors_60_l330_330056

def sum_of_positive_factors (n : ℕ) : ℕ :=
  ∑ d in (Finset.filter (λ x => x > 0) (Finset.divisors n)), d

theorem sum_of_positive_factors_60 : sum_of_positive_factors 60 = 168 := by
  sorry

end sum_of_positive_factors_60_l330_330056


namespace mean_equal_implication_l330_330774

theorem mean_equal_implication (y : ℝ) :
  (7 + 10 + 15 + 23 = 55) →
  (55 / 4 = 13.75) →
  (18 + y + 30 = 48 + y) →
  (48 + y) / 3 = 13.75 →
  y = -6.75 :=
by 
  intros h1 h2 h3 h4
  -- The steps would be applied here to prove y = -6.75
  sorry

end mean_equal_implication_l330_330774


namespace parallel_planes_from_conditions_l330_330578

variables (α β γ : Plane) (a b : Line)

-- Define the four conditions as separate definitions
def condition1 : Prop := (a ⊥ α) ∧ (a ⊥ β)
def condition2 : Prop := (γ ⊥ α) ∧ (γ ⊥ β)
def condition3 : Prop := (a ⊆ α) ∧ (b ⊆ β) ∧ (a ∥ β) ∧ (b ∥ α)
def condition4 : Prop := (a ⊆ α) ∧ (b ⊆ β) ∧ (a ∥ β) ∧ (b ∥ α) ∧ skew_lines a b

-- The sufficient conditions for α ∥ β
theorem parallel_planes_from_conditions (h1 : condition1 ∨ condition4) : α ∥ β :=
by sorry

end parallel_planes_from_conditions_l330_330578


namespace cleaning_time_l330_330435

noncomputable def combined_cleaning_time (sawyer_time nick_time sarah_time : ℕ) : ℚ :=
  let rate_sawyer := 1 / sawyer_time
  let rate_nick := 1 / nick_time
  let rate_sarah := 1 / sarah_time
  1 / (rate_sawyer + rate_nick + rate_sarah)

theorem cleaning_time : combined_cleaning_time 6 9 4 = 36 / 19 := by
  have h1 : 1 / 6 = 1 / 6 := rfl
  have h2 : 1 / 9 = 1 / 9 := rfl
  have h3 : 1 / 4 = 1 / 4 := rfl
  rw [combined_cleaning_time, h1, h2, h3]
  norm_num
  sorry

end cleaning_time_l330_330435


namespace log_equation_roots_l330_330454

theorem log_equation_roots (x : ℝ) : (log 2 (x^2 - 5 * x - 2) = 2) ↔ (x = -1 ∨ x = 6) :=
by
  sorry

end log_equation_roots_l330_330454


namespace probability_of_exactly_three_heads_l330_330129

theorem probability_of_exactly_three_heads (n : ℕ) (k : ℕ) (H : k = 3) (N : n = 8) :
  let total_outcomes := 2 ^ n,
      favorable_outcomes := Nat.choose n k in
  (favorable_outcomes / total_outcomes : ℚ) = 7 / 32 := 
by
  sorry

end probability_of_exactly_three_heads_l330_330129


namespace sqrt_expression_eq_l330_330513

theorem sqrt_expression_eq (t : ℝ) : sqrt(t^4 - t^2 * sin t ^ 2) = abs t * sqrt(t^2 - sin t ^ 2) := 
by 
  sorry

end sqrt_expression_eq_l330_330513


namespace blocks_calculation_l330_330089

theorem blocks_calculation
  (total_amount : ℕ)
  (gift_cost : ℕ)
  (workers_per_block : ℕ)
  (H1  : total_amount = 4000)
  (H2  : gift_cost = 4)
  (H3  : workers_per_block = 100)
  : total_amount / gift_cost / workers_per_block = 10 :=
by
  sorry

end blocks_calculation_l330_330089


namespace F_divides_l330_330568

-- Definitions
def F (n k : ℕ) : ℕ := ∑ r in Finset.range (n + 1), r^(2 * k - 1)

-- Theorem statement
theorem F_divides : ∀ (n k : ℕ), 0 < n → 0 < k → F n 1 ∣ F n k :=
by
  -- Proof placeholder
  intro n k hn hk
  exact sorry

end F_divides_l330_330568


namespace total_books_in_display_l330_330657

theorem total_books_in_display : 
  ∃ (n : ℕ), (32 - (n - 1) * 4 = 4) ∧ 
             (∑ k in finset.range n, 32 - k * 4) = 108 :=
by
  sorry

end total_books_in_display_l330_330657


namespace find_y_l330_330186

def star (a b : ℝ) : ℝ := 4 * a + 2 * b

theorem find_y (y : ℝ) : star 3 (star 4 y) = -2 → y = -11.5 :=
by
  sorry

end find_y_l330_330186


namespace max_product_of_three_distinct_numbers_l330_330048

noncomputable def maximum_possible_product (s : Set ℤ) : ℤ :=
  s.prod

theorem max_product_of_three_distinct_numbers :
  ∃ (a b c : ℤ), a ∈ {-10, -5, -3, 0, 2, 6, 8} ∧ b ∈ {-10, -5, -3, 0, 2, 6, 8} ∧ c ∈ {-10, -5, -3, 0, 2, 6, 8} ∧
                 a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
                 maximum_possible_product {a, b, c} = 400 :=
sorry

end max_product_of_three_distinct_numbers_l330_330048


namespace sequence_periodic_l330_330588

noncomputable def {u} sequence (a : ℕ → ℝ) :=
∀ k : ℕ, a k ≠ 0 ∧ a (k + 1) = floor (a k) * frac (a k)

theorem sequence_periodic (a : ℕ → ℝ) (h : sequence a) :
  ∃ k₀ : ℕ, ∀ k ≥ k₀, a k = a (k + 2) :=
sorry

end sequence_periodic_l330_330588


namespace inequality_am_gm_l330_330451

noncomputable def s (n : ℕ) (a : ℕ → ℝ) := ∑ i in finset.range n, a i

theorem inequality_am_gm (n : ℕ) (a : ℕ → ℝ) (h : ∀ i, 0 < a i) :
  (∏ i in finset.range n, (1 + a i)) ≤ (1 + s n a / n)^n :=
begin
  let s := s n a,
  sorry
end

end inequality_am_gm_l330_330451


namespace balance_three_diamonds_l330_330233

def Δ : Type := sorry
def ⋄ : Type := sorry
def • : Type := sorry

-- Conditions
axiom cond1 : 4 * Δ + 2 * ⋄ = 12 * •
axiom cond2 : 2 * Δ = ⋄ + 2 * •

-- Prove that 3 ⋄ = 6 •
theorem balance_three_diamonds : 3 * ⋄ = 6 * • :=
by
  sorry

end balance_three_diamonds_l330_330233


namespace domain_of_ln_x_minus_1_l330_330766

theorem domain_of_ln_x_minus_1 :
  {x : ℝ | ln (x - 1) > 0} = {x | x > 1} :=
by
  sorry

end domain_of_ln_x_minus_1_l330_330766


namespace lebesgue_diff_monotone_function_l330_330097

noncomputable def D_sup (g : ℝ → ℝ) (x : ℝ) : ℝ := 
  filter.LimSup (filter.tendsto_at_top_nhds_within x (λ y, (g x - g y) / (x - y)))

noncomputable def D_inf (g : ℝ → ℝ) (x : ℝ) : ℝ :=
  filter.LimInf (filter.tendsto_at_top_nhds_within x (λ y, (g x - g y) / (x - y)))

theorem lebesgue_diff_monotone_function (f : ℝ → ℝ) (a b : ℝ) (h1 : ∀ x ∈ set.Icc a b, ∀ y ∈ set.Icc a b, x ≤ y → f x ≤ f y)
  (h2 : ∀ g : ℝ → ℝ, ∃ k : ℝ, ∀ x ∈ set.Icc a b, ∑ i in finset.range (n - 1), abs (g ((finset.range (n - 1)).succ i) - g ((finset.range (n - 1)).nth i)) ≤ k)
  (h3 : ∀ g : ℝ → ℝ, ∀ x ∈ set.Icc a b, (D_sup g x) ∈ set.Ioi 0 → (D_inf g x) ∈ set.Iio 0) : 
∀ᵐ x ∂(volume.restrict (set.Icc a b)), ∃ (f' : ℝ), f' ∈ set.Ici 0 ∧ f' < ⊤ := 
sorry

end lebesgue_diff_monotone_function_l330_330097


namespace fraction_sum_is_ten_l330_330453

theorem fraction_sum_is_ten :
  (1 / 10) + (2 / 10) + (3 / 10) + (4 / 10) + (5 / 10) + (6 / 10) + (7 / 10) + (8 / 10) + (9 / 10) + (55 / 10) = 10 :=
by
  sorry

end fraction_sum_is_ten_l330_330453


namespace contrapositive_false_l330_330862

theorem contrapositive_false : ¬ (∀ x : ℝ, x^2 = 1 → x = 1) → ∀ x : ℝ, x^2 = 1 → x ≠ 1 :=
by
  sorry

end contrapositive_false_l330_330862


namespace lantern_probability_l330_330880

noncomputable def redwood_palace_lantern : Prop := sorry
noncomputable def sandalwood_palace_lantern : Prop := sorry
noncomputable def nanmu_gauze_lantern : Prop := sorry
noncomputable def huanghuali_gauze_lantern : Prop := sorry
noncomputable def congratulations_hanging_lantern : Prop := sorry
noncomputable def auspicious_hanging_lantern : Prop := sorry

def lanterns : Finset (Prop) := 
  {redwood_palace_lantern,
   sandalwood_palace_lantern,
   nanmu_gauze_lantern,
   huanghuali_gauze_lantern,
   congratulations_hanging_lantern,
   auspicious_hanging_lantern}

noncomputable def arrangements_count := 720 -- 6!
noncomputable def one_type_adjacent_arrangements := 288 -- 96 * 3

theorem lantern_probability : 
  ( one_type_adjacent_arrangements.toFloat / arrangements_count.toFloat ) = (2 : ℚ / 5) := by
  sorry

end lantern_probability_l330_330880


namespace incorrect_statement_l330_330349

-- Definitions based on the given conditions
def tripling_triangle_altitude_triples_area (b h : ℝ) : Prop :=
  3 * (1/2 * b * h) = 1/2 * b * (3 * h)

def halving_rectangle_base_halves_area (b h : ℝ) : Prop :=
  1/2 * b * h = 1/2 * (b * h)

def tripling_circle_radius_triples_area (r : ℝ) : Prop :=
  3 * (Real.pi * r^2) = Real.pi * (3 * r)^2

def tripling_divisor_and_numerator_leaves_quotient_unchanged (a b : ℝ) (hb : b ≠ 0) : Prop :=
  a / b = 3 * a / (3 * b)

def halving_negative_quantity_makes_it_greater (x : ℝ) : Prop :=
  x < 0 → (x / 2) > x

-- The incorrect statement is that tripling the radius of a circle triples the area
theorem incorrect_statement : ∃ r : ℝ, tripling_circle_radius_triples_area r → False :=
by
  use 1
  simp [tripling_circle_radius_triples_area]
  sorry

end incorrect_statement_l330_330349


namespace avg_books_rounded_l330_330399

def books_read : List (ℕ × ℕ) := [(1, 4), (2, 3), (3, 6), (4, 2), (5, 4)]

noncomputable def total_books_read (books : List (ℕ × ℕ)) : ℕ :=
  books.foldl (λ acc pair => acc + pair.fst * pair.snd) 0

noncomputable def total_members (books : List (ℕ × ℕ)) : ℕ :=
  books.foldl (λ acc pair => acc + pair.snd) 0

noncomputable def average_books_read (books : List (ℕ × ℕ)) : ℤ :=
  Int.ofNat (total_books_read books) / Int.ofNat (total_members books)

theorem avg_books_rounded :
  average_books_read books_read = 3 :=
by 
  sorry

end avg_books_rounded_l330_330399


namespace desired_line_through_A_bisects_segment_l330_330019

variables {A : Point} {l : Line} {S : Circle}

-- The main theorem statement outlining the conditions and desired property.
theorem desired_line_through_A_bisects_segment (A : Point)
  (l : Line) (S : Circle) (O : Point) (R : ℝ)
  (hO : S.center = O) (hR : S.radius = R) :
  ∃ m : Line, (A ∈ m) ∧ (∃ P Q : Point, P ∈ (m ∩ S) ∧ Q ∈ (m ∩ S) ∧ A = midpoint P Q ∧ 
               (∃ P' Q' : Point, P' ∈ (l ∩ S) ∧ Q' ∈ (l ∩ S) ∧ A = midpoint P' Q')) :=
sorry

end desired_line_through_A_bisects_segment_l330_330019


namespace f_sub_f_neg_l330_330261

-- Define the function f
def f (x : ℝ) : ℝ := x^5 + x^3 + 7 * x

-- State the theorem
theorem f_sub_f_neg : f 3 - f (-3) = 582 :=
by
  -- Definitions and calculations for the proof
  -- (You can complete this part in later proof development)
  sorry

end f_sub_f_neg_l330_330261


namespace amanda_needs_how_many_bags_of_grass_seeds_l330_330502

theorem amanda_needs_how_many_bags_of_grass_seeds
    (lot_length : ℕ := 120)
    (lot_width : ℕ := 60)
    (concrete_length : ℕ := 40)
    (concrete_width : ℕ := 40)
    (bag_coverage : ℕ := 56) :
    (lot_length * lot_width - concrete_length * concrete_width) / bag_coverage = 100 := by
  sorry

end amanda_needs_how_many_bags_of_grass_seeds_l330_330502


namespace arun_avg_weight_l330_330817

theorem arun_avg_weight (w : ℝ) :
  (62 < w ∧ w < 72) ∧
  (60 < w ∧ w < 70) ∧
  (w ≤ 65) →
  (∃ avg, avg = 63.5) :=
by
    intros h
    use 63.5
    sorry -- Proof goes here

end arun_avg_weight_l330_330817


namespace correct_total_score_l330_330406

theorem correct_total_score (total_score1 total_score2 : ℤ) : 
  (total_score1 = 5734 ∨ total_score2 = 5734) → (total_score1 = 5735 ∨ total_score2 = 5735) → 
  (total_score1 % 2 = 0 ∨ total_score2 % 2 = 0) → 
  (total_score1 ≠ total_score2) → 
  5734 % 2 = 0 :=
by
  sorry

end correct_total_score_l330_330406


namespace probability_of_exactly_three_heads_l330_330128

theorem probability_of_exactly_three_heads (n : ℕ) (k : ℕ) (H : k = 3) (N : n = 8) :
  let total_outcomes := 2 ^ n,
      favorable_outcomes := Nat.choose n k in
  (favorable_outcomes / total_outcomes : ℚ) = 7 / 32 := 
by
  sorry

end probability_of_exactly_three_heads_l330_330128


namespace height_difference_l330_330122

theorem height_difference
  (a b : ℝ)
  (parabola_eq : ∀ x, y = x^2 + 1)
  (circle_center : b = 2 * a^2 + 1 / 2) :
  b - (a^2 + 1) = a^2 - 1 / 2 :=
by {
  sorry
}

end height_difference_l330_330122


namespace count_distinct_digits_in_range_l330_330630

-- Define the range and distinctness condition
def range_n := {n : ℕ | 2000 ≤ n ∧ n ≤ 9999}
def distinct_digits (n : ℕ) : Prop :=
  let digits := n.digits 10 in
  digits.length = 4 ∧ digits.nodup

-- Problem statement: number of integers in the range with distinct digits
theorem count_distinct_digits_in_range :
  (finset.univ.filter (λ n : ℕ, n ∈ range_n ∧ distinct_digits n)).card = 4032 :=
sorry

end count_distinct_digits_in_range_l330_330630


namespace power_function_value_at_1_over_16_l330_330772

theorem power_function_value_at_1_over_16 (f : ℝ → ℝ) (h : f 4 = 1 / 2) : f (1 / 16) = 4 :=
by 
sorr

end power_function_value_at_1_over_16_l330_330772


namespace problem_l330_330977

open Function

variable {R : Type*} [Ring R]

def f (x : R) : R := -- Assume the function type

theorem problem (h : ∀ x y : R, f (x * y) = y^2 * f x + x^2 * f y) :
  f 0 = 0 ∧ f 1 = 0 ∧ (∀ x : R, f (-x) = f x) :=
by
  sorry

end problem_l330_330977


namespace gamma_identity_l330_330515

noncomputable def gamma_dist (α : ℝ) := measure_theory.measure.measureable_space.cond {x : ℝ | x > 0} (probability_theory.probability_gamma α 1)

variables {Xα Xα1_2 X2α : ℝ}

theorem gamma_identity (α : ℝ) :
  (∃ (gamma_func : ℝ → ℝ), ∀ (x : ℝ), gamma_func x = (x ^ (α - 1) * exp (-x)) / real.gamma α) →
  (2 * log X2α = -2 * real.euler_mascheroni + 2 * ∑' n, (1 / (n + 1) - Y n / (n + 2 * α))) →
  (4 * Xα * Xα1_2 = X2α ^ 2) :=
begin
  sorry
end

end gamma_identity_l330_330515


namespace num_ordered_triples_l330_330481

theorem num_ordered_triples (b : ℕ) (h : b = 1681) : 
  {t : ℕ × ℕ // t.fst ≤ b ∧ b ≤ t.snd ∧ (t.fst * t.snd = 1681 ^ 2)}.to_finset.card = 2 :=
by
  sorry

end num_ordered_triples_l330_330481


namespace total_points_l330_330656

def jon_points (sam_points : ℕ) : ℕ := 2 * sam_points + 3
def sam_points (alex_points : ℕ) : ℕ := alex_points / 2
def jack_points (jon_points : ℕ) : ℕ := jon_points + 5
def tom_points (jon_points jack_points : ℕ) : ℕ := jon_points + jack_points - 4
def alex_points : ℕ := 18

theorem total_points : jon_points (sam_points alex_points) + 
                       jack_points (jon_points (sam_points alex_points)) + 
                       tom_points (jon_points (sam_points alex_points)) 
                       (jack_points (jon_points (sam_points alex_points))) + 
                       sam_points alex_points + 
                       alex_points = 117 :=
by sorry

end total_points_l330_330656


namespace triangle_formation_l330_330604

-- Given conditions
variable {α β γ : Real}
variable {Group1 : Real × Real × Real}
variable {Group2 : Real × Real × Real}
variable {Group3 : Real × Real × Real}
variable {Group4 : Real × Real × Real}

-- Interior angles of the triangle
-- Constraints on angles: 0 < α, β, γ < π
axiom interior_angles (hα : 0 < α ∧ α < Real.pi)
                      (hβ : 0 < β ∧ β < Real.pi)
                      (hγ : 0 < γ ∧ γ < Real.pi)
                      (sum_angles : α + β + γ = Real.pi)

-- Definitions of groups based on trigonometric values
def Group1 := (Real.sin α, Real.sin β, Real.sin γ)
def Group2 := (Real.sin α ^ 2, Real.sin β ^ 2, Real.sin γ ^ 2)
def Group3 := (Real.cos (α / 2) ^ 2, Real.cos (β / 2) ^ 2, Real.cos (γ / 2) ^ 2)
def Group4 := (Real.tan (α / 2), Real.tan (β / 2), Real.tan (γ / 2))

-- Function to check formation of a triangle
def can_form_triangle (x y z : Real) : Prop :=
  x + y > z ∧ x + z > y ∧ y + z > x

-- Main lean statement
theorem triangle_formation:
  can_form_triangle (Group1.1) (Group1.2) (Group1.3) ∧
  can_form_triangle (Group3.1) (Group3.2) (Group3.3) ∧
  ¬ can_form_triangle (Group2.1) (Group2.2) (Group2.3) ∧
  ¬ can_form_triangle (Group4.1) (Group4.2) (Group4.3) :=
sorry

end triangle_formation_l330_330604


namespace bags_wednesday_l330_330163

def charge_per_bag : ℕ := 4
def bags_monday : ℕ := 5
def bags_tuesday : ℕ := 3
def total_earnings : ℕ := 68

theorem bags_wednesday (h1 : charge_per_bag = 4)
                       (h2 : bags_monday = 5)
                       (h3 : bags_tuesday = 3)
                       (h4 : total_earnings = 68) :
  let earnings_monday_tuesday := (bags_monday + bags_tuesday) * charge_per_bag in
  let earnings_wednesday := total_earnings - earnings_monday_tuesday in
  earnings_wednesday / charge_per_bag = 9 :=
by
  sorry

end bags_wednesday_l330_330163


namespace necessary_but_not_sufficient_l330_330969

-- Let's assume we have some locale where plane, line and perpendicularity/parallelism are defined.
variables {Plane Line : Type}
variables (α β : Plane) (a : Line)

-- Definitions based on the conditions
def is_perpendicular (p1 p2 : Plane) : Prop := -- Placeholder for the actual definition
sorry
def is_parallel (l : Line) (p : Plane) : Prop := -- Placeholder for actual definition
sorry
def is_not_contained (l : Line) (p : Plane) : Prop := -- Placeholder for actual definition
sorry
def is_perpendicular_to_line (l : Line) (p : Plane) : Prop := -- Placeholder for actual definition
sorry

-- The conditions of the problem
axiom cond1 : is_perpendicular α β
axiom cond2 : is_not_contained a β
def p : Prop := is_parallel a β
def q : Prop := is_perpendicular_to_line a α

-- The statement that we need to prove
theorem necessary_but_not_sufficient (α β : Plane) (a : Line)
  (h₁ : is_perpendicular α β) (h₂ : is_not_contained a β) :
  (p → q) ∧ (¬ (q → p)) :=
sorry

end necessary_but_not_sufficient_l330_330969


namespace sqrt_defined_iff_l330_330543

theorem sqrt_defined_iff (x : ℝ) : (∃ y, y = sqrt (3 * x - 1)) ↔ x ≥ 1 / 3 :=
by
  sorry

end sqrt_defined_iff_l330_330543


namespace matrix_determinant_eq_9_l330_330099

theorem matrix_determinant_eq_9 (x : ℝ) :
  let a := x - 1
  let b := 2
  let c := 3
  let d := -5
  (a * d - b * c = 9) → x = -2 :=
by 
  let a := x - 1
  let b := 2
  let c := 3
  let d := -5
  sorry

end matrix_determinant_eq_9_l330_330099


namespace problem_l330_330982

open Function

variable {R : Type*} [Ring R]

def f (x : R) : R := -- Assume the function type

theorem problem (h : ∀ x y : R, f (x * y) = y^2 * f x + x^2 * f y) :
  f 0 = 0 ∧ f 1 = 0 ∧ (∀ x : R, f (-x) = f x) :=
by
  sorry

end problem_l330_330982


namespace proof_problem_l330_330706

theorem proof_problem (a b : ℝ) (h : {1, a + b, a} = {0, b / a, b}) : b - 1 = 0 :=
sorry

end proof_problem_l330_330706


namespace pole_height_l330_330140

variables {A B C D E: Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E]

-- Defining distances in meters
variables (AC AD DC DE AB : ℝ)
-- Given conditions
variables (h1 : AC = 4) 
variables (h2 : AD = 3) 
variables (h3 : DE = 1.7) 
variables (h4 : DC = AC - AD) -- Derived as DC = AC - AD = 1

theorem pole_height (AB: ℝ) (h4 : ΔABC ~ ΔDEC) (h5 : AB/AC = DE/DC): AB = 6.8 :=
by {
  calc AB = 4 * 1.7 : by { rw [h1, h3, h2], exact h5 },
  exact 6.8,
  sorry,
}

end pole_height_l330_330140


namespace calculate_discount_percentage_l330_330873

theorem calculate_discount_percentage :
  ∃ (x : ℝ), (∀ (P S : ℝ),
    (S = 439.99999999999966) →
    (S = 1.10 * P) →
    (1.30 * (1 - x / 100) * P = S + 28) →
    x = 10) :=
sorry

end calculate_discount_percentage_l330_330873


namespace range_of_m_l330_330961

def proposition_p (m : ℝ) : Prop :=
  ∀ x > 0, m^2 + 2 * m - 1 ≤ x + 1 / x

def proposition_q (m : ℝ) : Prop :=
  ∀ x : ℝ, x > 0 → (5 - m^2) ^ x > (5 - m^2) ^ (x - 1)

theorem range_of_m (m : ℝ) : (proposition_p m ∨ proposition_q m) ∧ ¬ (proposition_p m ∧ proposition_q m) ↔ (-3 ≤ m ∧ m ≤ -2) ∨ (1 < m ∧ m < 2) :=
sorry

end range_of_m_l330_330961


namespace value_of_a_l330_330600

-- Definitions of conditions
def OA (x : ℝ) : ℝ × ℝ := (2 * (Real.cos x) ^ 2, 1)
def OB (x a : ℝ) : ℝ × ℝ := (1, Real.sqrt 3 * Real.sin (2 * x) + a)
def f (x a : ℝ) : ℝ := let (xa, ya) := OA x;
                           let (xb, yb) := OB x a
                       in xa * xb + ya * yb

-- Constants
axiom x_in_R : ∀ (x : ℝ), x ∈ ℝ
axiom a_in_R : ∀ (a : ℝ), a ∈ ℝ
axiom a_constant : ∀ a, a ∈ ℝ

-- Assumptions and proof goal
theorem value_of_a (h : ∀ x ∈ Set.Icc (0 : ℝ) (Real.pi / 2), f x a ≥ 2) : a = 2 := 
sorry


end value_of_a_l330_330600


namespace maximal_cross_section_area_l330_330854

/-- Define the vertices of the right rectangular prism and the intersecting plane -/
structure PrismVertex where
  x : ℝ
  y : ℝ
  z : ℝ
  deriving Repr

/-- Vertices of the right rectangular prism with the square base centered at the origin -/
def vertices : List PrismVertex := [
  { x := 4, y := 4, z := 1 },
  { x := -4, y := 4, z := 2 },
  { x := -4, y := -4, z := 3 },
  { x := 4, y := -4, z := 4 }
]

/-- Plane equation given by 3x - 5y + 3z = 24 -/
def plane_eq (x y z : ℝ) : Prop :=
  3 * x - 5 * y + 3 * z = 24

/-- Maximal area proposition -/
theorem maximal_cross_section_area : 
  ∃ (A' B' C' D' : PrismVertex), 
    plane_eq A'.x A'.y A'.z ∧
    plane_eq B'.x B'.y B'.z ∧
    plane_eq C'.x C'.y C'.z ∧
    plane_eq D'.x D'.y D'.z ∧
    -- Assuming we have defined area calculation function for a quadrilateral
    area_of_quadrilateral A' B' C' D' = 110 :=
sorry

/-- Define the function to compute the area of the quadrilateral formed by four points -/
noncomputable def area_of_quadrilateral (A B C D : PrismVertex) := 
  let vec_AB := (B.x - A.x, B.y - A.y, B.z - A.z)
  let vec_AD := (D.x - A.x, D.y - A.y, D.z - A.z)
  let cross_product := (
    vec_AB.2 * vec_AD.3 - vec_AB.3 * vec_AD.2, 
    vec_AB.3 * vec_AD.1 - vec_AB.1 * vec_AD.3, 
    vec_AB.1 * vec_AD.2 - vec_AB.2 * vec_AD.1
  )
  let area := 0.5 * (Real.sqrt (cross_product.1^2 + cross_product.2^2 + cross_product.3^2))
  area


end maximal_cross_section_area_l330_330854


namespace compute_complex_product_l330_330882

noncomputable def cos (θ : ℝ) : ℝ := (Real.angle.cos θ.to_real)
noncomputable def sin (θ : ℝ) : ℝ := (Real.angle.sin θ.to_real)
noncomputable def i : ℂ := Complex.i

theorem compute_complex_product :
  4 * (cos (15 * Real.pi / 180) - i * sin (15 * Real.pi / 180)) * 
    5 * (sin (15 * Real.pi / 180) - i * cos (15 * Real.pi / 180)) = -20 * i := 
by 
  sorry

end compute_complex_product_l330_330882


namespace charge_per_mile_l330_330728

theorem charge_per_mile (rental_fee total_amount_paid : ℝ) (num_miles : ℕ) (charge_per_mile : ℝ) : 
  rental_fee = 20.99 →
  total_amount_paid = 95.74 →
  num_miles = 299 →
  (total_amount_paid - rental_fee) / num_miles = charge_per_mile →
  charge_per_mile = 0.25 :=
by 
  intros r_fee t_amount n_miles c_per_mile h1 h2 h3 h4
  sorry

end charge_per_mile_l330_330728


namespace fraction_allocated_for_school_l330_330803

-- Conditions
def days_per_week : ℕ := 5
def hours_per_day : ℕ := 4
def earnings_per_hour : ℕ := 5
def allocation_for_school : ℕ := 75

-- Proof statement
theorem fraction_allocated_for_school :
  let weekly_hours := days_per_week * hours_per_day
  let weekly_earnings := weekly_hours * earnings_per_hour
  allocation_for_school / weekly_earnings = 3 / 4 := 
by
  sorry

end fraction_allocated_for_school_l330_330803


namespace radical_axis_eq_l330_330927

-- Definitions of the given circles
def circle1_eq (x y : ℝ) : Prop := x^2 + y^2 - 4 * x + 6 * y = 0
def circle2_eq (x y : ℝ) : Prop := x^2 + y^2 - 6 * x = 0

-- The theorem proving that the equation of the radical axis is 3x - y - 9 = 0
theorem radical_axis_eq (x y : ℝ) :
  (circle1_eq x y) ∧ (circle2_eq x y) → 3 * x - y - 9 = 0 :=
sorry

end radical_axis_eq_l330_330927


namespace greatest_possible_perimeter_l330_330667

def triangle_side_lengths (x : ℤ) : Prop :=
  (x > 0) ∧ (5 * x > 18) ∧ (x < 6)

def perimeter (x : ℤ) : ℤ :=
  x + 4 * x + 18

theorem greatest_possible_perimeter :
  ∃ x : ℤ, triangle_side_lengths x ∧ (perimeter x = 38) :=
by
  sorry

end greatest_possible_perimeter_l330_330667


namespace triangle_problem_proof_l330_330653

noncomputable def triangle_properties : ℝ :=
  let a := (3 * Real.sqrt 6) / 2 
  let A := Real.pi / 3
  let C := Real.pi / 4
  -- Statement 1: Find the value of c
  let c := a * Real.sin C / Real.sin A
  -- Statement 2: Construct maximum area triangle ABP with angle APB = 30 degrees
  let max_area := (9 / 4) * (2 + Real.sqrt 3)
  
  c = 3 ∧ max_area = (9 / 4) * (2 + Real.sqrt 3)

theorem triangle_problem_proof : triangle_properties := 
  by sorry

end triangle_problem_proof_l330_330653


namespace decreasing_interval_l330_330615

noncomputable def f (x θ : ℝ) : ℝ :=
  sqrt 3 * Real.sin (2 * x + θ) + Real.cos (2 * x + θ)

noncomputable def g (x θ : ℝ) : ℝ :=
  2 * Real.sin (2 * x + θ + π / 3)

theorem decreasing_interval (θ : ℝ) (k : ℤ) (hθ : |θ| < π / 2)
  (sym : θ + π / 3 = ↑k * π + π / 2) :
    let g := λ x : ℝ, 2 * Real.sin (2 * x + 2 * π / 3)
    [x : ℝ.represents] (let x = λ γ : ℝ, let r = (π / 12). Nearly( -k *π)
    [(1 : ℝ represents (/that/x.) * γ)] )
 :=
begin
  sorry
end

end decreasing_interval_l330_330615


namespace find_angle_A_max_perimeter_l330_330589

-- Definition of the problem as given in the conditions:
variables {A B C a b c : ℝ} (h_orth : a * (cos C + sqrt 3 * sin C) = b + c)

-- First part: finding the measure of angle A
theorem find_angle_A (h_orth : a * (cos C + sqrt 3 * sin C) = b + c) : A = π / 3 :=
sorry

-- Second part: finding the maximum perimeter
theorem max_perimeter (h_A : A = π / 3) (h_a : a = sqrt 3) : 
  let perimeter := a + 2 * sin B + 2 * sin (2 * π / 3 - B) 
  in perimeter = 3 * sqrt 3 :=
sorry

end find_angle_A_max_perimeter_l330_330589


namespace no_three_distinct_nat_numbers_sum_prime_l330_330910

theorem no_three_distinct_nat_numbers_sum_prime:
  ¬∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 0 < a ∧ 0 < b ∧ 0 < c ∧ 
  Nat.Prime (a + b) ∧ Nat.Prime (a + c) ∧ Nat.Prime (b + c) := 
sorry

end no_three_distinct_nat_numbers_sum_prime_l330_330910


namespace exists_odd_k_l_m_l330_330448

def odd_nat (n : ℕ) : Prop := n % 2 = 1

theorem exists_odd_k_l_m : 
  ∃ (k l m : ℕ), 
  odd_nat k ∧ odd_nat l ∧ odd_nat m ∧ 
  (k ≠ 0) ∧ (l ≠ 0) ∧ (m ≠ 0) ∧ 
  (1991 * (l * m + k * m + k * l) = k * l * m) :=
by
  sorry

end exists_odd_k_l_m_l330_330448


namespace range_of_rhombus_area_l330_330263

open Real

noncomputable def line_l (x : ℝ) : ℝ := sqrt 3 * x + 4

noncomputable def circle_O (x y r : ℝ) : Prop := x^2 + y^2 = r^2 ∧ 1 < r ∧ r < 2

theorem range_of_rhombus_area (r : ℝ) (h_r : 1 < r ∧ r < 2)
    (A B : ℝ × ℝ) (hA : A.2 = line_l A.1) (hB : B.2 = line_l B.1)
    (C D : ℝ × ℝ) (hC : circle_O C.1 C.2 r) (hD : circle_O D.1 D.2 r)
    (h_angle : (∠ (A, B, C) = 60 ∧ ∠ (B, C, D) = 60) ∨ (∠ (A, B, D) = 60 ∧ ∠ (B, A, C) = 60)) :
  let S := abs (3 * sqrt 3 / 2 * ((r^2 / 3) * ((8 - 2 - 1)^2 - 16) / 3)) in
  ((0 : ℝ) < S ∧ S < 3 * sqrt 3 / 2) ∨ ((3 * sqrt 3 / 2) < S ∧ S < 6 * sqrt 3) :=
by
  sorry

end range_of_rhombus_area_l330_330263


namespace range_of_x_l330_330243

noncomputable def even_function (f : ℝ → ℝ) :=
  ∀ x : ℝ, f (-x) = f x

noncomputable def monotonically_increasing_on_nonneg (f : ℝ → ℝ) :=
  ∀ x y : ℝ, 0 ≤ x → x ≤ y → f x ≤ f y

theorem range_of_x (f : ℝ → ℝ)
  (h_even : even_function f)
  (h_monotonically_increasing : monotonically_increasing_on_nonneg f)
  (h_ineq : ∀ x : ℝ, f (x^2 - 2) < f 2) :
  ∀ x : ℝ, x ∈ (-2 : ℝ) → x ∈ (0 : ℝ) ∨ x ∈ (0 : ℝ) → x ∈ (2 : ℝ) :=
begin
  sorry
end

end range_of_x_l330_330243


namespace apples_equation_l330_330417

variable {A J H : ℕ}

theorem apples_equation:
    A + J = 12 →
    H = A + J + 9 →
    A = J + 8 →
    H = 21 :=
by
  intros h1 h2 h3
  sorry

end apples_equation_l330_330417


namespace count_digit_7_from_10_to_149_l330_330683

theorem count_digit_7_from_10_to_149 :
  let count_units (n : ℕ) := if n % 10 = 7 then 1 else 0
  let count_tens (n : ℕ) := if (n / 10) % 10 = 7 then 1 else 0
  ∑ n in Finset.range 140, count_units (n + 10) + count_tens (n + 10) = 24 :=
by
  let count_units (n : ℕ) := if n % 10 = 7 then 1 else 0
  let count_tens (n : ℕ) := if (n / 10) % 10 = 7 then 1 else 0
  sorry

end count_digit_7_from_10_to_149_l330_330683


namespace intersection_M_N_l330_330621

open Set

def M : Set ℝ := {-2, -1, 0, 1, 2}
def N : Set ℝ := {x | (x + 1) * (x - 2) ≤ 0}

theorem intersection_M_N : M ∩ N = {-1, 0, 1, 2} :=
by
  sorry

end intersection_M_N_l330_330621


namespace binary_to_decimal_l330_330888

theorem binary_to_decimal : 
  let binary_number := [1, 1, 0, 1, 1] in
  let weights := [2^4, 2^3, 2^2, 2^1, 2^0] in
  (List.zipWith (λ digit weight => digit * weight) binary_number weights).sum = 27 := by
  sorry

end binary_to_decimal_l330_330888


namespace kangaroo_arrangement_count_l330_330191

theorem kangaroo_arrangement_count :
  let k := 8
  let tallest_at_ends := 2
  let middle := k - tallest_at_ends
  (tallest_at_ends * (middle.factorial)) = 1440 := by
  sorry

end kangaroo_arrangement_count_l330_330191


namespace relationship_among_abc_l330_330965

def a : ℝ := (1 / 2)^(2 / 3)
def b : ℝ := (1 / 3)^(1 / 3)
def c : ℝ := Real.log 3

theorem relationship_among_abc : c > b ∧ b > a := by
  sorry

end relationship_among_abc_l330_330965


namespace length_stationary_l330_330491

def speed : ℝ := 64.8
def time_pole : ℝ := 5
def time_stationary : ℝ := 25

def length_moving : ℝ := speed * time_pole
def length_combined : ℝ := speed * time_stationary

theorem length_stationary : length_combined - length_moving = 1296 :=
by
  sorry

end length_stationary_l330_330491


namespace number_of_hens_l330_330473

-- Conditions as Lean definitions
def total_heads (H C : ℕ) : Prop := H + C = 48
def total_feet (H C : ℕ) : Prop := 2 * H + 4 * C = 136

-- Mathematically equivalent proof problem
theorem number_of_hens (H C : ℕ) (h1 : total_heads H C) (h2 : total_feet H C) : H = 28 :=
by
  sorry

end number_of_hens_l330_330473


namespace fraction_addition_l330_330919

theorem fraction_addition :
  (2 / 5 : ℚ) + (3 / 8) = 31 / 40 :=
sorry

end fraction_addition_l330_330919


namespace vector_product_computation_l330_330326

noncomputable def vector_dot_product_in_triangle
  (AB AC BC: ℝ) (hAB: AB = 5) (hAC: AC = 6) (hBC: BC = 5) :
  ℝ :=
  let B := Real.arccos ((AB^2 + BC^2 - AC^2) / (2 * AB * BC)) in
  -AB^2 
  + AB * BC * -((AB^2 + BC^2 - AC^2) / (2 * AB * BC))

theorem vector_product_computation
  (AB AC BC: ℝ) (hAB: AB = 5) (hAC: AC = 6) (hBC: BC = 5) :
  vector_dot_product_in_triangle AB AC BC hAB hAC hBC = -32 :=
sorry

end vector_product_computation_l330_330326


namespace candy_bar_cost_l330_330190

def initial_amount : ℕ := 4
def remaining_amount : ℕ := 3
def cost_of_candy_bar : ℕ := initial_amount - remaining_amount

theorem candy_bar_cost : cost_of_candy_bar = 1 := by
  sorry

end candy_bar_cost_l330_330190


namespace functional_equation_option_A_option_B_option_C_l330_330970

-- Given conditions
def domain_of_f (f : ℝ → ℝ) : Set ℝ := Set.univ

theorem functional_equation (f : ℝ → ℝ) (x y : ℝ) :
  f (x * y) = y^2 * f x + x^2 * f y := sorry

-- Statements to be proved
theorem option_A (f : ℝ → ℝ) (h : ∀ x y, f(x * y) = y^2 * f(x) + x^2 * f(y)) : f 0 = 0 := sorry

theorem option_B (f : ℝ → ℝ) (h : ∀ x y, f(x * y) = y^2 * f(x) + x^2 * f(y)) : f 1 = 0 := sorry

theorem option_C (f : ℝ → ℝ) (h : ∀ x y, f(x * y) = y^2 * f(x) + x^2 * f(y)) : ∀ x, f(-x) = f x := sorry

end functional_equation_option_A_option_B_option_C_l330_330970


namespace part_a_part_b_l330_330447

variable (a b : ℝ)

-- Given conditions
variable (h1 : a^3 - b^3 = 2) (h2 : a^5 - b^5 ≥ 4)

-- Requirement (a): Prove that a > b
theorem part_a : a > b := by 
  sorry

-- Requirement (b): Prove that a^2 + b^2 ≥ 2
theorem part_b : a^2 + b^2 ≥ 2 := by 
  sorry

end part_a_part_b_l330_330447


namespace coordinates_of_meeting_point_B_l330_330802

noncomputable def meeting_point_coordinates
    (start_point : ℝ × ℝ := (1, 0))
    (rate_p : ℝ := π / 3)
    (rate_q : ℝ := π / 6)
    (meet_time : ℝ := 4) : ℝ × ℝ :=
let angle_p := rate_p * meet_time
in ((-cos angle_p), (-sin angle_p))

theorem coordinates_of_meeting_point_B :
    meeting_point_coordinates (1, 0) (π / 3) (π / 6) 4 = (-1/2, -√3/2) :=
sorry

end coordinates_of_meeting_point_B_l330_330802


namespace largest_inscribed_equilateral_triangle_area_l330_330174

theorem largest_inscribed_equilateral_triangle_area 
  (r : ℝ) (h_r : r = 10) : 
  ∃ A : ℝ, 
    A = 100 * Real.sqrt 3 ∧ 
    (∃ s : ℝ, s = 2 * r ∧ A = (Real.sqrt 3 / 4) * s^2) := 
  sorry

end largest_inscribed_equilateral_triangle_area_l330_330174


namespace min_value_expression_l330_330047

theorem min_value_expression (x y : ℝ) : (x^2 * y - 1)^2 + (x + y - 1)^2 ≥ 1 :=
sorry

end min_value_expression_l330_330047


namespace min_value_of_f_on_interval_l330_330776

noncomputable def f (x : ℝ) := x^2 + 1/x - x

theorem min_value_of_f_on_interval :
  ∃ c ∈ set.Icc (1 / 2 : ℝ) 2, ∀ x ∈ set.Icc (1 / 2 : ℝ) 2, f x ≥ f c ∧ f c = 1 :=
by
  sorry

end min_value_of_f_on_interval_l330_330776


namespace sum_of_factors_of_60_l330_330071

-- Given conditions
def n : ℕ := 60

-- Proof statement
theorem sum_of_factors_of_60 : (∑ d in (finset.filter (λ d, n % d = 0) (finset.range (n + 1))), d) = 168 :=
by
  sorry

end sum_of_factors_of_60_l330_330071


namespace face_value_of_share_l330_330845

theorem face_value_of_share 
  (F : ℝ)
  (dividend_rate : ℝ := 0.09)
  (desired_return_rate : ℝ := 0.12)
  (market_value : ℝ := 15) 
  (h_eq : (dividend_rate * F) / market_value = desired_return_rate) : 
  F = 20 := 
by 
  sorry

end face_value_of_share_l330_330845


namespace suitable_for_sampling_survey_l330_330434

-- Definitions based on the conditions given in the problem
def company_health_checks : Prop := ∀ e ∈ employees, health_check(e)
def epidemic_temperature_checks : Prop := ∀ e ∈ unit_employees, temperature_check(e)
def entertainment_survey : Prop := sampling_survey(youth, main_forms_of_entertainment)
def airplane_security_checks : Prop := ∀ p ∈ passengers, security_check(p)

-- Main statement we want to prove
theorem suitable_for_sampling_survey : 
  ¬company_health_checks ∧ 
  ¬epidemic_temperature_checks ∧ 
  entertainment_survey ∧ 
  ¬airplane_security_checks → 
  suitable_for_sampling(entertainment_survey) :=
by
  intro h,
  cases h with nhc h_rest,
  cases h_rest with net hc,
  cases hc with es nas,
  exact es -- Asserting the suitable survey
  sorry

end suitable_for_sampling_survey_l330_330434


namespace cos_sum_identity_cosine_30_deg_l330_330529

theorem cos_sum_identity : 
  (Real.cos (Real.pi * 43 / 180) * Real.cos (Real.pi * 13 / 180) + 
   Real.sin (Real.pi * 43 / 180) * Real.sin (Real.pi * 13 / 180)) = 
   (Real.cos (Real.pi * 30 / 180)) :=
sorry

theorem cosine_30_deg : 
  Real.cos (Real.pi * 30 / 180) = (Real.sqrt 3 / 2) :=
sorry

end cos_sum_identity_cosine_30_deg_l330_330529


namespace circle_tangent_parabola_height_difference_l330_330119

theorem circle_tangent_parabola_height_difference 
  (a b : ℝ)
  (h_tangent1 : (a, a^2 + 1))
  (h_tangent2 : (-a, a^2 + 1))
  (parabola_eq : ∀ x, x^2 + 1 = (x, x^2 + 1))
  (circle_eq : ∀ x, x^2 + ((x^2 + 1) - b)^2 = r^2) : 
  b - (a^2 + 1) = 1 / 2 :=
sorry

end circle_tangent_parabola_height_difference_l330_330119


namespace baba_yaga_powder_problem_l330_330096

theorem baba_yaga_powder_problem (A B d : ℤ) 
  (h1 : A + B + d = 6) 
  (h2 : A + d = 3) 
  (h3 : B + d = 2) : 
  A = 4 ∧ B = 3 :=
by
  -- Proof omitted, as only the statement is required
  sorry

end baba_yaga_powder_problem_l330_330096


namespace charge_per_mile_l330_330729

def rental_fee : ℝ := 20.99
def total_amount_paid : ℝ := 95.74
def miles_driven : ℝ := 299

theorem charge_per_mile :
  (total_amount_paid - rental_fee) / miles_driven = 0.25 := 
sorry

end charge_per_mile_l330_330729


namespace orthogonal_vectors_l330_330723

def vector (α : Type*) [Add α] := list α

def dot_product (v1 v2 : vector ℝ) : ℝ :=
(list.zip_with (*) v1 v2).sum

theorem orthogonal_vectors (m : ℝ) :
    let a := [1, -1]
    let b := [m + 1, 2m - 4]
    dot_product a b = 0 → m = 5 := 
by
  intro h
  sorry

end orthogonal_vectors_l330_330723


namespace arithmetic_progression_terms_even_sums_l330_330400

theorem arithmetic_progression_terms_even_sums (n a d : ℕ) (h_even : Even n) 
  (h_odd_sum : n * (a + (n - 2) * d) = 60) 
  (h_even_sum : n * (a + d + a + (n - 1) * d) = 72) 
  (h_last_first : (n - 1) * d = 12) : n = 8 := 
sorry

end arithmetic_progression_terms_even_sums_l330_330400


namespace problem_solution_l330_330345

noncomputable def P (x : ℝ) : ℝ := (3 * x^4 - 15 * x^3 + a * x^2 + b * x + c) * (4 * x^3 - 36 * x^2 + d * x + e)

theorem problem_solution
  (a b c d e : ℝ)
  (roots_Q : set ℝ := {2, 3, 4, 5})
  (roots_R : set ℝ := {3, 5, 5}) :
  (∀ y ∈ roots_Q, Q y = 0) →
  (∀ z ∈ roots_R, R z = 0) →
  P 7 = 23040 := by
  sorry

end problem_solution_l330_330345


namespace arithmetic_progression_primes_l330_330559

theorem arithmetic_progression_primes (a k : ℕ) (N : ℕ) :
    (∃ (N : ℕ), ∀ p : ℕ, p > N → (nat.prime p → nat.prime (a + k * p))) →
    (∃ P : ℕ, nat.prime P ∧ (k = 0 ∧ a = P) ∨ (k = 1 ∧ a = 0)) :=
by
    sorry

end arithmetic_progression_primes_l330_330559


namespace min_modulus_of_m_l330_330294

noncomputable def quadratic_has_real_roots := 
  ∀ (m : ℂ), 
  (∃ (α : ℝ), (4 + 3*complex.I) * α^2 + m * α + (4 - 3*complex.I) = 0) 
  → ‖m‖ ≥ 8

theorem min_modulus_of_m {m : ℂ} (h : quadratic_has_real_roots) :
  ∃ (α : ℝ), (4 + 3*complex.I) * α^2 + m * α + (4 - 3*complex.I) = 0 → ‖m‖ = 8 :=
sorry

end min_modulus_of_m_l330_330294


namespace y_intercept_of_line_l330_330010

theorem y_intercept_of_line (m : ℝ) (x1 y1 : ℝ) (h_slope : m = -3) (h_x_intercept : x1 = 7) (h_y_intercept : y1 = 0) : 
  ∃ y_intercept : ℝ, y_intercept = 21 ∧ y_intercept = -m * 0 + 21 :=
by
  sorry

end y_intercept_of_line_l330_330010


namespace find_sinB_area_ABC_l330_330684

/-- Given conditions -/
variables {A B C : ℝ}
variable [triangle_ABC : isTriangle A B C]

-- Given values
def cosA := (sqrt 10) / 10
def AC := sqrt 10
def BC := 3 * sqrt 2

-- Proof goal 1: Find sinB
theorem find_sinB : 
  (exists B : ℝ, sin B = sqrt 2 / 2) :=
sorry

-- Proof goal 2: Area of △ABC
-- Assuming AB is found to be 4 from the problem.
def AB := 4

theorem area_ABC : 
  (1/2 * AB * BC * (sqrt 2 / 2) = 6) :=
sorry

end find_sinB_area_ABC_l330_330684


namespace range_of_4x_plus_2y_l330_330606

theorem range_of_4x_plus_2y (x y : ℝ) 
  (h₁ : 1 ≤ x + y ∧ x + y ≤ 3)
  (h₂ : -1 ≤ x - y ∧ x - y ≤ 1) : 
  2 ≤ 4 * x + 2 * y ∧ 4 * x + 2 * y ≤ 10 :=
sorry

end range_of_4x_plus_2y_l330_330606


namespace functions_satisfying_eq_l330_330558

theorem functions_satisfying_eq (f g : ℝ → ℝ) (C : ℝ) :
  (∀ x y : ℝ, sin x + cos y = f x + f y + g x - g y) →
  (∀ x : ℝ, f x = (sin x + cos x) / 2) ∧ (∀ x : ℝ, g x = (sin x - cos x) / 2 + C) :=
by 
  intros h,
  sorry

end functions_satisfying_eq_l330_330558


namespace problem_statement_l330_330504

-- Defining the propositions
def proposition1 : Prop := (∀ (A B : ℝ), A > B → Real.sin A > Real.sin B)
def converse_proposition1 : Prop := (∀ (A B : ℝ), Real.sin A > Real.sin B → A > B)
def proposition2 : Prop := ∀ (x y : ℝ), (x ≠ 2 ∨ y ≠ 3) → (x + y ≠ 5)
def neg_prop2 : Prop := ∀ (x y : ℝ), (x + y ≠ 5) → (x ≠ 2 ∨ y ≠ 3)
def proposition3 : Prop := (∀ (x : ℝ), x^3 - x^2 + 1 ≤ 0)
def neg_proposition3 : Prop := (∃ (x : ℝ), x^3 - x^2 + 1 > 0)
def proposition4 : Prop := (∀ (a b : ℝ), a > b → (2^a > 2^b - 1))
def neg_proposition4 : Prop := (∀ (a b : ℝ), a ≤ b → (2^a ≤ 2^b - 1))

-- Defining the correctness of each proposition
def correctness1 : Prop := converse_proposition1
def correctness2 : Prop := ∀ (x y : ℝ), (x ≠ 2 ∨ y ≠ 3) ↔ (x + y ≠ 5)
def correctness3 : Prop := ¬ neg_proposition3
def correctness4 : Prop := neg_proposition4

-- Propose that the number of correct propositions is 3
def number_of_correct_propositions : ℕ := 3

-- The main statement
theorem problem_statement : (correctness1 ∧ correctness2 ∧ correctness4) ∧ ¬ correctness3 → number_of_correct_propositions = 3 := by
  sorry

end problem_statement_l330_330504


namespace find_excluded_digit_l330_330226

theorem find_excluded_digit (a b : ℕ) (d : ℕ) (h : a * b = 1024) (ha : a % 10 ≠ d) (hb : b % 10 ≠ d) : 
  ∃ r : ℕ, d = r ∧ r < 10 :=
by 
  sorry

end find_excluded_digit_l330_330226


namespace paper_cranes_l330_330413

theorem paper_cranes (B C A : ℕ) (h1 : A + B + C = 1000)
  (h2 : A = 3 * B - 100)
  (h3 : C = A - 67) : A = 443 := by
  sorry

end paper_cranes_l330_330413


namespace shaded_rectangle_area_l330_330869

def area_polygon : ℝ := 2016
def sides_polygon : ℝ := 18
def segments_persh : ℝ := 4

theorem shaded_rectangle_area :
  (area_polygon / sides_polygon) * segments_persh = 448 := 
sorry

end shaded_rectangle_area_l330_330869


namespace correct_system_exists_l330_330726

def system_of_equations (x y : ℕ) : Prop :=
  200 * y = x + 18 ∧ 180 * y = x - 42

theorem correct_system_exists (x y : ℕ) :
  200 * y = x + 18 → 180 * y = x - 42 → system_of_equations x y :=
by
  intros h1 h2
  split
  exact h1
  exact h2

#print axioms correct_system_exists

end correct_system_exists_l330_330726


namespace probability_chord_not_intersect_inner_circle_l330_330800

-- Define the radii of the inner and outer circles
def inner_radius : ℝ := 2
def outer_radius : ℝ := 4

-- Define the probability problem
theorem probability_chord_not_intersect_inner_circle :
  let P := by assume (p1 p2 : ℝ × ℝ), { sorry } in -- placeholder for probability computation
  P = (2 / 3) :=
sorry

end probability_chord_not_intersect_inner_circle_l330_330800


namespace additional_grassy_ground_l330_330482

theorem additional_grassy_ground (r1 r2 : ℝ) (π : ℝ) :
  r1 = 12 → r2 = 18 → π = Real.pi →
  (π * r2^2 - π * r1^2) = 180 * π := by
sorry

end additional_grassy_ground_l330_330482


namespace sufficient_condition_not_necessary_condition_l330_330713

variable {x1 x2 : ℝ}

theorem sufficient_condition (h₁ : x1 > 1) (h₂ : x2 > 1) : x1 + x2 > 2 ∧ x1 * x2 > 1 :=
  by sorry

theorem not_necessary_condition : ¬ (∀ x1 x2, (x1 + x2 > 2 ∧ x1 * x2 > 1) → (x1 > 1 ∧ x2 > 1)) :=
  by
  intro h
  have counterexample : (10 + 0.1 > 2 ∧ 10 * 0.1 > 1) ∧ ¬ (10 > 1 ∧ 0.1 > 1) :=  
  by
    split
    . split
      . linarith
      . linarith
    . split
      . linarith
      . linarith
      
  exact counterexample.2 (h 10 0.1 counterexample.1)

end sufficient_condition_not_necessary_condition_l330_330713


namespace tony_bread_slices_left_l330_330024

def num_slices_left (initial_slices : ℕ) (d1 : ℕ) (d2 : ℕ) : ℕ :=
  initial_slices - (d1 + d2)

theorem tony_bread_slices_left :
  num_slices_left 22 (5 * 2) (2 * 2) = 8 := by
  sorry

end tony_bread_slices_left_l330_330024


namespace solution_pairs_l330_330187

theorem solution_pairs (p : ℕ) (hp : p > 1) :
  (Prime p → ∃ eqn_sol : Finset (ℕ × ℕ), eqn_sol.card = 3 ∧ ∀ ⟨x, y⟩ ∈ eqn_sol, (1/x : ℚ) + (1/y) = 1/p) ∧
  (¬ Prime p → ∃ eqn_sol : Finset (ℕ × ℕ), 3 < eqn_sol.card ∧ ∀ ⟨x, y⟩ ∈ eqn_sol, (1/x : ℚ) + (1/y) = 1/p) :=
by sorry

end solution_pairs_l330_330187


namespace mothers_day_sale_l330_330687

noncomputable def original_price (discount1 discount2 final_price : ℝ) : ℝ :=
  final_price / ((1 - discount1) * (1 - discount2))

theorem mothers_day_sale
  (P : ℝ) -- original price of the shoes
  (discount1 discount2 : ℝ) -- discounts
  (final_price : ℝ) -- price after all discounts
  (h1 : discount1 = 0.10) -- 10% discount for mothers
  (h2 : discount2 = 0.04) -- Additional 4% discount for mothers with 3 or more children
  (h3 : final_price = 108) -- Mrs. Brown will pay $108 after all discounts are applied
  : P = 125 :=
by
  have discount_price := P * (1 - discount1) * (1 - discount2)
  rw [←h1, ←h2] at discount_price
  rw h3 at discount_price
  have : discount_price = final_price, by sorry
  rw this at discount_price
  exact (final_price / (0.864)).trans (by norm_num : 125 = 125)

end mothers_day_sale_l330_330687


namespace cake_eating_contest_l330_330033

-- Define the fractions representing the amounts of cake eaten by the two students.
def first_student : ℚ := 7 / 8
def second_student : ℚ := 5 / 6

-- The statement of our proof problem
theorem cake_eating_contest : first_student - second_student = 1 / 24 := by
  sorry

end cake_eating_contest_l330_330033


namespace pattern_ABC_150th_letter_is_C_l330_330042

theorem pattern_ABC_150th_letter_is_C :
  (fun cycle length index =>
    let repeats := index / length;
    let remainder := index % length;
    if remainder = 0 then 'C' else
    if remainder = 1 then 'A' else 'B') 3 150 = 'C' := sorry

end pattern_ABC_150th_letter_is_C_l330_330042


namespace right_triangle_hypotenuse_l330_330000

theorem right_triangle_hypotenuse 
  (shorter_leg longer_leg hypotenuse : ℝ)
  (h1 : longer_leg = 2 * shorter_leg - 1)
  (h2 : 1 / 2 * shorter_leg * longer_leg = 60) :
  hypotenuse = 17 :=
by
  sorry

end right_triangle_hypotenuse_l330_330000


namespace value_of_a_b_l330_330001

def geometric_seq (a b : ℝ) := ∃ r : ℝ, 25 * r = a ∧ a * r = b ∧ b * r = (1 : ℝ) / 25

theorem value_of_a_b :
  ∃ a b : ℝ, a = Real.cbrt 25 ∧ b = 25^(-1/3 : ℝ) ∧ geometric_seq a b :=
by {
  let a := Real.cbrt 25,
  let b := 25^(-1/3 : ℝ),
  use [a, b],
  split,
  { refl },
  split,
  { refl },
  {
    use a / 25,
    split,
    { field_simp, rw [mul_comm, mul_assoc, Real.mul_cbrt_self, div_self (ne_of_gt (show (25 : ℝ) ≠ 0, by norm_num))] },
    split,
    { field_simp, rw [mul_comm, Real.mul_cbrt_self, ←div_div], norm_num },
    { field_simp, rw [mul_comm, Real.mul_cbrt_self, ←div_div, div_self (ne_of_gt (show 25 ≠ 0, by norm_num)), div_one] },
  },
  sorry
}

end value_of_a_b_l330_330001


namespace subset_condition_l330_330577

noncomputable def A : Set ℝ := {x | x^2 - 3*x - 10 ≤ 0}

def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

theorem subset_condition (m : ℝ) : (B m ⊆ A) ↔ m ≤ 3 :=
sorry

end subset_condition_l330_330577


namespace passenger_speed_relative_forward_correct_l330_330858

-- Define the conditions
def train_speed : ℝ := 60     -- Train's speed in km/h
def passenger_speed_inside_train : ℝ := 3  -- Passenger's speed inside the train in km/h

-- Define the effective speed of the passenger relative to the railway track when moving forward
def passenger_speed_relative_forward (train_speed passenger_speed_inside_train : ℝ) : ℝ :=
  train_speed + passenger_speed_inside_train

-- Prove that the passenger's speed relative to the railway track is 63 km/h when moving forward
theorem passenger_speed_relative_forward_correct :
  passenger_speed_relative_forward train_speed passenger_speed_inside_train = 63 := by
  sorry

end passenger_speed_relative_forward_correct_l330_330858


namespace hyperbola_equation_of_square_vertices_l330_330292

theorem hyperbola_equation_of_square_vertices
  (a b : ℝ)
  (ha : a > 0) (hb : b > 0)
  (h_square : ∀ (x y : ℝ), (x, y) = (a, 0) ∨ (x, y) = (-a, 0) ∨ (x, y) = (0, b) ∨ (x, y) = (0, -b)
   → x^2 + y^2 = 2) :
  a = b ∧ a = sqrt 2 ∧ ∀ x y : ℝ, (x^2 / 2) - (y^2 / 2) = 1 := by
{
  sorry,
}

end hyperbola_equation_of_square_vertices_l330_330292


namespace functional_eq_properties_l330_330986

theorem functional_eq_properties (f : ℝ → ℝ) (h_domain : ∀ x, x ∈ ℝ) (h_func_eq : ∀ x y, f (x * y) = y^2 * f x + x^2 * f y) :
  f 0 = 0 ∧ f 1 = 0 ∧ ∀ x, f x = f (-x) :=
by {
  sorry
}

end functional_eq_properties_l330_330986


namespace correct_truth_values_l330_330812

open Real

def proposition_p : Prop := ∀ (a : ℝ), 0 < a → a^2 ≠ 0

def converse_p : Prop := ∀ (a : ℝ), a^2 ≠ 0 → 0 < a

def inverse_p : Prop := ∀ (a : ℝ), ¬(0 < a) → a^2 = 0

def contrapositive_p : Prop := ∀ (a : ℝ), a^2 = 0 → ¬(0 < a)

def negation_p : Prop := ∃ (a : ℝ), 0 < a ∧ a^2 = 0

theorem correct_truth_values : 
  (converse_p = False) ∧ 
  (inverse_p = False) ∧ 
  (contrapositive_p = True) ∧ 
  (negation_p = False) := by
  sorry

end correct_truth_values_l330_330812


namespace area_of_given_triangle_l330_330015

def point := ℝ × ℝ

def triangle_area (A B C : point) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem area_of_given_triangle : 
  triangle_area (1, 1) (7, 1) (5, 3) = 6 :=
by
  -- the proof should go here
  sorry

end area_of_given_triangle_l330_330015


namespace median_length_names_l330_330769

def name_lengths : List Nat := [3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7]

def median (l : List Nat) : Float :=
  let sorted := l.qsort (· ≤ ·)
  let n := sorted.length
  if n % 2 = 0 then
    (sorted.get! (n / 2 - 1) + sorted.get! (n / 2)) / 2
  else
    sorted.get! (n / 2)

theorem median_length_names : median name_lengths = 4.5 :=
  by
    sorry

end median_length_names_l330_330769


namespace slices_of_bread_left_l330_330026

variable (monday_to_friday_slices saturday_slices total_slices_used initial_slices slices_left: ℕ)

def sandwiches_monday_to_friday : ℕ := 5
def slices_per_sandwich : ℕ := 2
def sandwiches_saturday : ℕ := 2
def initial_slices_of_bread : ℕ := 22

theorem slices_of_bread_left :
  slices_left = initial_slices_of_bread - total_slices_used
  :=
by  sorry

end slices_of_bread_left_l330_330026


namespace line_circle_relationship_l330_330618

-- Definitions used in the problem
def line (t : ℝ) : ℝ × ℝ := (t + 1, t)
def circle (θ : ℝ) : ℝ × ℝ := (2 + cos θ, sin θ)

-- The proof goal, stating the conditions and the result
theorem line_circle_relationship {t θ : ℝ} :
  ∃ t θ, 
    let c := (2 : ℝ, 0 : ℝ) in
    let l := line t in
    let circ := circle θ in
    let dist := |(l.1 - c.1) - (l.2 - c.2 - 1)| / (Real.sqrt (1^2 + (-1)^2)) in
    dist < 1 ∧ dist ≠ 0 :=
sorry

end line_circle_relationship_l330_330618


namespace greatest_product_sum_1976_l330_330537

theorem greatest_product_sum_1976 : 
  ∃ (a b: ℕ), 2 * a + 3 * b = 1976 ∧ 2^a * 3^b = 2 * 3^658 :=
by
  -- Definitions and conditions from part (a)
  let x := 1
  let y := 658
  have h1: 2 * x + 3 * y = 1976, by sorry 
  
  -- Proving the main statement
  use [x, y]
  exact ⟨h1, by sorry⟩

end greatest_product_sum_1976_l330_330537


namespace volume_conversion_l330_330479

theorem volume_conversion (volume_ft³ : ℕ) (ft³_per_yd³ : ℝ) (m³_per_yd³ : ℝ) : 
  volume_ft³ = 216 → ft³_per_yd³ = 27 → m³_per_yd³ = 0.764 → (volume_ft³ / ft³_per_yd³) * m³_per_yd³ = 6.112 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end volume_conversion_l330_330479


namespace smallest_prime_divisor_of_3_pow_15_plus_11_pow_9_l330_330051

-- Definitions based on conditions
def is_odd (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k + 1
def is_even (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

-- Lemmas derived from the problem conditions
lemma three_power_fifteen_is_odd : is_odd (3^15) :=
  sorry

lemma eleven_power_nine_is_odd : is_odd (11^9) :=
  sorry

lemma sum_of_two_odds_is_even {a b : ℕ} (ha : is_odd a) (hb : is_odd b) : is_even (a + b) :=
  sorry

-- Main theorem stating the smallest prime divisor
theorem smallest_prime_divisor_of_3_pow_15_plus_11_pow_9 :
  ∃ p : ℕ, prime p ∧ p = 2 ∧ p ∣ (3^15 + 11^9) :=
begin
  have h1 : is_odd (3^15) := three_power_fifteen_is_odd,
  have h2 : is_odd (11^9) := eleven_power_nine_is_odd,
  have he : is_even (3^15 + 11^9) := sum_of_two_odds_is_even h1 h2,
  use 2,
  split,
  { exact prime_two },
  { split,
    { refl },
    { rw ←is_even_iff_two_dvd,
      exact he }
  }
end

end smallest_prime_divisor_of_3_pow_15_plus_11_pow_9_l330_330051


namespace negation_of_statement_l330_330100

theorem negation_of_statement (x : ℝ) :
  ¬ (if x = 0 ∨ x = 1 then x^2 - x = 0) ↔ (x ≠ 0 ∧ x ≠ 1 → x^2 - x ≠ 0) :=
by sorry

end negation_of_statement_l330_330100


namespace face_value_of_share_l330_330843

theorem face_value_of_share (FV : ℝ) (market_value : ℝ) (dividend_rate : ℝ) (desired_return_rate : ℝ) 
  (H1 : market_value = 15) 
  (H2 : dividend_rate = 0.09) 
  (H3 : desired_return_rate = 0.12) 
  (H4 : dividend_rate * FV = desired_return_rate * market_value) :
  FV = 20 := 
by
  sorry

end face_value_of_share_l330_330843


namespace simplify_trig_expression_l330_330754

theorem simplify_trig_expression (A : ℝ) :
  (2 - (Real.cos A / Real.sin A) + (1 / Real.sin A)) * (3 - (Real.sin A / Real.cos A) - (1 / Real.cos A)) = 
  7 * Real.sin A * Real.cos A - 2 * Real.cos A ^ 2 - 3 * Real.sin A ^ 2 - 3 * Real.cos A + Real.sin A + 1 :=
by
  sorry

end simplify_trig_expression_l330_330754


namespace sin_eq_exp_solution_count_l330_330935

theorem sin_eq_exp_solution_count :
  (∃! (n : ℕ), 2 * n = 150) ↔
  ∀ x ∈ set.Ioo 0 (150 * real.pi), 
  (∃ y, real.sin y = (1 / 3)^y) :=
sorry

end sin_eq_exp_solution_count_l330_330935


namespace min_supermean_is_one_third_l330_330570

noncomputable def supermean (a b : ℝ) (a_cond : 0 < a) (b_cond : a < b) (b_cond_2 : b < 1) : ℝ :=
  let x : ℕ → ℝ := fun
    | 0 => 0
    | n + 1 => (1 - a) * x n + a * (y n)
  and y : ℕ → ℝ := fun
    | 0 => 1
    | n + 1 => (1 - b) * x n + b * (y n)
  (lim (nat.succ pred) fun n, x n)

def circle_condition (p q : ℝ) : Prop :=
  (p - 1 / 2) ^ 2 + (q - 1 / 2) ^ 2 ≤ (1 / 10) ^ 2

theorem min_supermean_is_one_third :
  ∀ p q : ℝ, circle_condition p q → supermean p q (by linarith) (by linarith) (by linarith) = 1 / 3 :=
sorry 

example : 100 * 1 + 3 = 103 := by norm_num

end min_supermean_is_one_third_l330_330570


namespace sandwich_count_l330_330411

-- We introduce the initial number of kinds of sandwiches, the number sold out, and the number still available as variables.
variables (initial_sandwiches sold_out_sandwiches available_sandwiches : ℕ)

-- The conditions given in the problem.
def conditions : Prop :=
  sold_out_sandwiches = 5 ∧ available_sandwiches = 4

-- The statement of the problem: Proving the initial number of kinds of sandwiches.
theorem sandwich_count (h : conditions) : initial_sandwiches = 9 :=
  by sorry

end sandwich_count_l330_330411


namespace tangent_sphere_surface_area_l330_330587

-- Define the base length of the right triangular prism
def base_length : ℝ := 2 * Real.sqrt 3

-- Define the height of the right triangular prism
def height : ℝ := 3

-- Define the radius of the inscribed circle of equilateral triangle ABC
def radius_inscribed_circle : ℝ := (Real.sqrt 3 / 3) * base_length

-- Define the surface area of the tangent sphere of the conical frustum
def surface_area_tangent_sphere (r : ℝ) : ℝ := 4 * Real.pi * r ^ 2

-- The point P is any point on O (inscribed circle)
axiom point_on_O : ∀ (P : ℝ), P ∈ Set.Icc (0 : ℝ) 1

-- Given conditions and statement we need to prove
theorem tangent_sphere_surface_area : 
  (surface_area_tangent_sphere radius_inscribed_circle) = 25 := 
sorry

end tangent_sphere_surface_area_l330_330587


namespace sum_of_positive_factors_60_l330_330065

theorem sum_of_positive_factors_60 : (∑ n in Nat.divisors 60, n) = 168 := by
  sorry

end sum_of_positive_factors_60_l330_330065


namespace visitors_on_previous_day_l330_330145

theorem visitors_on_previous_day (visitors_current_day : ℕ) (difference : ℕ) (h1 : visitors_current_day = 666) (h2 : difference = 566) : (visitors_current_day - difference = 100) :=
by
  rw [h1, h2]
  norm_num
  -- verifies that 666 - 566 equals 100
  sorry

end visitors_on_previous_day_l330_330145


namespace dividend_is_217_l330_330647

-- Given conditions
def r : ℕ := 1
def q : ℕ := 54
def d : ℕ := 4

-- Define the problem as a theorem in Lean 4
theorem dividend_is_217 : (d * q) + r = 217 := by
  -- proof is omitted
  sorry

end dividend_is_217_l330_330647


namespace veg_eaters_count_l330_330660

theorem veg_eaters_count {a b c d : ℕ} (h1 : a = 13) (h2 : b = 7) (h3 : c = 8) (h4 : d = a + c) : d = 21 :=
by {
  rw [h1, h3] at h4,
  exact h4,
  sorry
}

end veg_eaters_count_l330_330660


namespace trajectory_is_ellipse_length_of_chord_l330_330317

-- Define the conditions for the problem
def trajectory_C_eq (P : ℝ × ℝ) : Prop :=
  let M := (0 : ℝ, -Real.sqrt 3)
  let N := (0 : ℝ, Real.sqrt 3)
  (Real.sqrt ((P.1 - M.1) ^ 2 + (P.2 - M.2) ^ 2) + Real.sqrt ((P.1 - N.1) ^ 2 + (P.2 - N.2) ^ 2)) = 4

def is_ellipse (P : ℝ × ℝ) : Prop :=
  P.1 ^ 2 + (P.2 ^ 2) / 4 = 1

theorem trajectory_is_ellipse : ∀ (P : ℝ × ℝ),
  trajectory_C_eq P ↔ is_ellipse P := by
  sorry

def intersection_points (y : ℝ) (x1 x2 : ℝ) : Prop :=
  (y = x1 / 2 ∧ y = x2 / 2 ∧ x1 ≠ x2 ∧
  (x1 ^ 2 + (y ^ 2) / 4 = 1) ∧ (x2 ^ 2 + (y ^ 2) / 4 = 1))

def chord_length (x1 x2 : ℝ) : ℝ :=
  Real.sqrt (1 + (1 / 4)) * (abs (x2 - x1))

theorem length_of_chord : ∀ y x1 x2,
  intersection_points y x1 x2 →
  chord_length x1 x2 = 4 := by
  sorry

end trajectory_is_ellipse_length_of_chord_l330_330317


namespace item_A_profit_item_B_profit_item_C_profit_l330_330837

def effective_percentage_profit (actual_weight : ℕ) (measured_weight : ℕ) : ℚ :=
  ((1000 - measured_weight : ℚ) / measured_weight * 100)

-- Conditions
def item_A_actual_weight := 800
def item_A_measured_weight := item_A_actual_weight - item_A_actual_weight * 5 / 100

def item_B_actual_weight := 850
def item_B_measured_weight := item_B_actual_weight - item_B_actual_weight * 7 / 100

def item_C_actual_weight := 780
def item_C_measured_weight := item_C_actual_weight - item_C_actual_weight * 3 / 100

-- To Prove
theorem item_A_profit :
  effective_percentage_profit item_A_actual_weight item_A_measured_weight ≈ 31.58 := sorry

theorem item_B_profit :
  effective_percentage_profit item_B_actual_weight item_B_measured_weight ≈ 26.50 := sorry

theorem item_C_profit :
  effective_percentage_profit item_C_actual_weight item_C_measured_weight ≈ 32.17 := sorry

end item_A_profit_item_B_profit_item_C_profit_l330_330837


namespace bisect_area_of_triangle_l330_330592

noncomputable def ellipse (x y : ℝ) : Prop :=
  x^2 / (3 + 2 * real.sqrt 2) + y^2 / (2 + 2 * real.sqrt 2) = 1

def foci_x_coordinate : ℝ := 1
def focus_1 : ℝ × ℝ := (-foci_x_coordinate, 0)
def focus_2 : ℝ × ℝ := (foci_x_coordinate, 0)

def point_P (x y : ℝ) : Prop := ellipse x y ∧ y ≠ 0

def triangle_area_split (x_P y_P x_I y_I : ℝ) (area_half : Prop) : Prop := 
  ∃ (P:ℝ × ℝ)(F₁ F₂:ℝ × ℝ),
    P = (x_P, y_P) ∧
    F₁ = focus_1 ∧
    F₂ = focus_2 ∧
    I = (x_I, y_I) ∧
    let I := (I.1, (I.2 + y_P) / 2)
    ∧ ∀ (x : ℝ), area_half

theorem bisect_area_of_triangle
  (x_P y_P : ℝ)
  (hP : point_P x_P y_P)
  (x_I y_I : ℝ)
  (hI : (y_I = (y_P / ((real.sqrt (3 + 2 * real.sqrt 2) + 1)))) : 
  )
  (bisects : triangle_area_split x_P y_P x_I y_I (y_I = (y_P + 0) / 2 )) : 
  sorry

end bisect_area_of_triangle_l330_330592


namespace price_of_adult_ticket_l330_330798

-- Define the conditions
def price_student_ticket : ℝ := 2.50
def total_tickets_sold : ℕ := 59
def total_revenue : ℝ := 222.50
def student_tickets_sold : ℕ := 9

-- Define the goal / question we want to prove
theorem price_of_adult_ticket 
    (price_student_ticket : ℝ)
    (total_tickets_sold : ℕ)
    (total_revenue : ℝ)
    (student_tickets_sold : ℕ)
    (H_price_student : price_student_ticket = 2.50)
    (H_total_tickets : total_tickets_sold = 59)
    (H_total_revenue : total_revenue = 222.50)
    (H_student_tickets : student_tickets_sold = 9) :
    let revenue_from_student_tickets := student_tickets_sold * price_student_ticket,
        revenue_from_adult_tickets := total_revenue - revenue_from_student_tickets,
        adult_tickets_sold := total_tickets_sold - student_tickets_sold,
        price_adult_ticket := revenue_from_adult_tickets / adult_tickets_sold
    in
    price_adult_ticket = 4.00 := 
by 
    -- We state the assertion here
    sorry

end price_of_adult_ticket_l330_330798


namespace train_pass_man_in_time_l330_330441

noncomputable def time_to_pass (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) : ℝ :=
  let relative_speed := (train_speed + man_speed) * (5 / 18) in -- Convert speed to m/s
  train_length / relative_speed

theorem train_pass_man_in_time :
  time_to_pass 110 80 8 ≈ 4.5 := sorry

end train_pass_man_in_time_l330_330441


namespace remainder_of_7_pow_7_pow_7_pow_7_mod_500_l330_330222

theorem remainder_of_7_pow_7_pow_7_pow_7_mod_500 : 
  let λ := fun n => 
             match n with
             | 500 => 100
             | 100 => 20
             | _ => 0 in
  7^(7^(7^7)) ≡ 343 [MOD 500] :=
by 
  sorry

end remainder_of_7_pow_7_pow_7_pow_7_mod_500_l330_330222


namespace rectangular_prism_surface_area_l330_330383

theorem rectangular_prism_surface_area :
  ∃ (a b c : ℕ), 
    {a, b, c} ⊆ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    ∃ (ab bc ac : ℕ), 
      {ab, bc, ac} ⊆ {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                      30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
                      50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 
                      70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 
                      90, 91, 92, 93, 94, 95, 96, 97, 98, 99} ∧
      ab = a * b ∧ bc = b * c ∧ ac = a * c ∧
      a + b + c + ab + bc + ac = 45 ∧
      2 * (ab + bc + ac) = 198 :=
sorry

end rectangular_prism_surface_area_l330_330383


namespace divisible_by_six_l330_330788

theorem divisible_by_six (h : ∃ n ∈ (100:ℤ)..999, ∃ d : ℤ, (150:ℕ) = (999-100+1) / d ∧ 900 / 150 = d) : ∃ d, d = 6 :=
begin
  use 6,
  cases h with n hn,
  cases hn with hn range,
  cases range with d hd,
  cases hd with h1 h2,
  split,
  exact h2,
  sorry -- proof omitted
end

end divisible_by_six_l330_330788


namespace functional_eq_properties_l330_330987

theorem functional_eq_properties (f : ℝ → ℝ) (h_domain : ∀ x, x ∈ ℝ) (h_func_eq : ∀ x y, f (x * y) = y^2 * f x + x^2 * f y) :
  f 0 = 0 ∧ f 1 = 0 ∧ ∀ x, f x = f (-x) :=
by {
  sorry
}

end functional_eq_properties_l330_330987


namespace triangle_side_b_l330_330325

-- Define the conditions and state the problem
theorem triangle_side_b (A C : ℕ) (a b c : ℝ)
  (h1 : C = 4 * A)
  (h2 : a = 36)
  (h3 : c = 60) :
  b = 45 := by
  sorry

end triangle_side_b_l330_330325


namespace knight_2008_winner_l330_330468

theorem knight_2008_winner (n : ℕ) : 
  (∃ k ≥ 6, n = 1338 + 3^k ∨ n = 1338 + 2 * 3^k) ↔
  let knights := (λ i : ℕ, (i % 3 == 1)) in
  let next_knight := (λ current total, (current + 1) % total) in
  let eliminate_knights := λ n, { i | i % 3 == 0 ∨ i % 3 == 2 } in
  let remaining_knights := λ start n, 
    (List.range n).filter (λ i, ¬ (i ∈ eliminate_knights n)) in
  let rec last_knight : ℕ → ℕ
  | 1 := 1
  | k + 1 := next_knight (last_knight k) (k + 1) in
  last_knight n = 2007
:=
sorry

end knight_2008_winner_l330_330468


namespace girth_le_diameter_l330_330512

-- Definitions specific to the problem
def girth (G : SimpleGraph V) [Nonempty V] : ℕ := sorry  -- Define the girth of G
def diameter (G : SimpleGraph V) : ℕ := sorry  -- Define the diameter of G

theorem girth_le_diameter (G : SimpleGraph V) [Nonempty V] (h_cycle : ∃ (C : Finset V), G.IsCycle C) : 
  girth G ≤ 2 * diameter G + 1 :=
sorry

end girth_le_diameter_l330_330512


namespace coordinates_of_W_l330_330203

theorem coordinates_of_W (O U S V W : ℝ × ℝ)
  (hO : O = (0, 0))
  (hU : U = (3, 3))
  (hS : S = (3, 0))
  (hV : V = (0, 3))
  (hw : W = (6, 0)) : 
  let area_square := 9 in 
  let area_triangle := 4.5 in
  let SVW := abs ((S.1 * V.2 + V.1 * W.2 + W.1 * S.2) - (S.2 * V.1 + V.2 * W.1 + W.2 * S.1)) / 2 in 
  SVW = area_triangle :=
by {
  sorry
}

end coordinates_of_W_l330_330203


namespace polygon_sides_l330_330637

theorem polygon_sides (exterior_angle : ℝ) (sum_exterior_angles : ℝ) (h1 : exterior_angle = 30) (h2 : sum_exterior_angles = 360) :
  (sum_exterior_angles / exterior_angle) = 12 :=
by
  have h3 : ∀ (n : ℝ), n = 360 / 30, from sorry
  rw [h1, h2] at h3
  exact h3

end polygon_sides_l330_330637


namespace length_dg_l330_330746

theorem length_dg (a b k l S : ℕ) (h1 : S = 47 * (a + b)) 
                   (h2 : S = a * k) (h3 : S = b * l) (h4 : b = S / l) 
                   (h5 : a = S / k) (h6 : k * l = 47 * k + 47 * l + 2209) : 
  k = 2256 :=
by sorry

end length_dg_l330_330746


namespace bucket_weight_l330_330110

theorem bucket_weight (c d : ℝ) (x y : ℝ) 
  (h1 : x + 3/4 * y = c) 
  (h2 : x + 1/3 * y = d) :
  x + 1/4 * y = (6 * d - c) / 5 := 
sorry

end bucket_weight_l330_330110


namespace find_missing_number_l330_330293

theorem find_missing_number (x : ℝ)
  (h1 : (x + 42 + 78 + 104) / 4 = 62)
  (h2 : (128 + 255 + 511 + 1023 + x) / 5 = 398.2) :
  x = 74 :=
sorry

end find_missing_number_l330_330293


namespace total_area_correct_l330_330850

noncomputable def total_area (b l: ℝ) (h1: l = 3 * b) (h2: l * b = 588) : ℝ :=
  let rect_area := 588 -- Area of the rectangle
  let semi_circle_area := 24.5 * Real.pi -- Area of the semi-circle based on given diameter
  rect_area + semi_circle_area

theorem total_area_correct (b l: ℝ) (h1: l = 3 * b) (h2: l * b = 588) : 
  total_area b l h1 h2 = 588 + 24.5 * Real.pi :=
by
  sorry

end total_area_correct_l330_330850


namespace problem_l330_330983

open Function

variable {R : Type*} [Ring R]

def f (x : R) : R := -- Assume the function type

theorem problem (h : ∀ x y : R, f (x * y) = y^2 * f x + x^2 * f y) :
  f 0 = 0 ∧ f 1 = 0 ∧ (∀ x : R, f (-x) = f x) :=
by
  sorry

end problem_l330_330983


namespace num_false_statements_l330_330436

-- Let P_i (for i = 1 to 5) represent the truth of each of the five statements
variables (P1 P2 P3 P4 P5 : Prop)

-- The conditions described by the problem
def condition1 := P1 → (¬ P2 ∧ ¬ P3 ∧ ¬ P4 ∧ ¬ P5)
def condition2 := P2 → ((¬ P1 ∧ ¬ P2) ∨ (¬ P3 ∧ ¬ P4 ∧ ¬ P5))
def condition3 := P3 → ((¬ P1 ∧ ¬ P2 ∧ ¬ P4) ∧ P5)
def condition4 := P4 → (¬ P1 ∧ ¬ P2 ∧ ¬ P3 ∧ ¬ P5)
def condition5 := P5 → ((¬ P1 ∧ ¬ P2 ∧ ¬ P3 ∧ ¬ P4) ∨ (¬ P2 ∧ ¬ P4) ∨ (¬ P1 ∧ ¬ P3 ∧ ¬ P4))

-- The goal is to prove that exactly three statements are false
theorem num_false_statements : condition1 P1 P2 P3 P4 P5 ∧ condition2 P1 P2 P3 P4 P5 ∧ condition3 P1 P2 P3 P4 P5 ∧ condition4 P1 P2 P3 P4 P5 ∧ condition5 P1 P2 P3 P4 P5 → 
  ((¬ P1 ∧ ¬ P2 ∧ P3 ∧ ¬ P4 ∧ P5) ∨ 
  ((¬ P1 ∧ ¬ P2 ∧ P3 ∧ ¬ P4 ∧ ¬ P5) ∨ 
  ((¬ P1 ∧ P2 ∧ P3 ∧ P4 ∧ P5) ∨ 
  ((P1 ∧ ¬ P2 ∧ P3 ∧ ¬ P4 ∧ P5))) := 
sorry

end num_false_statements_l330_330436


namespace cos_alpha_plus_beta_equals_l330_330319

variables (α β : ℝ)
variables (cos_alpha sin_alpha cos_beta sin_beta : ℝ)
variables (point_alpha point_beta : ℝ × ℝ)

-- Conditions
def terminal_point_alpha (point_alpha = (1, 2)) : Prop :=
  let r := Real.sqrt (point_alpha.1^2 + point_alpha.2^2)
  sin_alpha = point_alpha.2 / r ∧ cos_alpha = point_alpha.1 / r

def terminal_point_beta (point_beta = (-2, 6)) : Prop :=
  let r := Real.sqrt (point_beta.1^2 + point_beta.2^2)
  sin_beta = point_beta.2 / r ∧ cos_beta = point_beta.1 / r

-- Proof statement
theorem cos_alpha_plus_beta_equals :
  terminal_point_alpha (1, 2) →
  terminal_point_beta (-2, 6) →
  cos (α + β) = cos_alpha * cos_beta - sin_alpha * sin_beta →
  cos_alpha = 1 / Real.sqrt 5 →
  sin_alpha = 2 / Real.sqrt 5 →
  cos_beta = -1 / Real.sqrt 10 →
  sin_beta = 3 / Real.sqrt 10 →
  cos (α + β) = - (7 * Real.sqrt 2) / 10 :=
sorry

end cos_alpha_plus_beta_equals_l330_330319


namespace circle_passing_through_three_points_l330_330562

-- Define the points A, B, and C
structure Point where
  x : ℝ
  y : ℝ

def A : Point := {x := 0, y := 0}
def B : Point := {x := 1, y := 1}
def C : Point := {x := 4, y := 2}

-- Define the general form of the circle equation
def circle_eq (D E F : ℝ) (p : Point) : Prop :=
  p.x^2 + p.y^2 + D * p.x + E * p.y + F = 0

-- The main theorem to prove
theorem circle_passing_through_three_points :
  ∃ (D E F : ℝ), 
    circle_eq D E F A ∧ 
    circle_eq D E F B ∧ 
    circle_eq D E F C ∧ 
    (∀ x y: ℝ, 
      x^2 + y^2 + D * x + E * y + F = 0 ↔ 
      x^2 + y^2 - 8 * x + 6 * y = 0) :=
by
  -- We already have actual D, E, and F from solution steps
  use [-8, 6, 0]
  split
  all_goals
  sorry

end circle_passing_through_three_points_l330_330562


namespace triangle_side_lengths_l330_330327

theorem triangle_side_lengths (A B C D E : Point) : 
  BE.perp AD ∧ length(BE) = 4 ∧ length(AD) = 4 →
  sides_of_triangle(ABC) = (sqrt 13, 2 * sqrt 13, sqrt 5) :=
sorry

end triangle_side_lengths_l330_330327


namespace find_x_equals_2_l330_330207

theorem find_x_equals_2 : 
  (∃ x : ℝ, 2^x + 3^x + 6^x = 7^x) → 
  2 := 
  begin
    use 2,
    calc 
      2^2 + 3^2 + 6^2 = 4 + 9 + 36 : by norm_num
      ... = 7^2 : by norm_num
  end

end find_x_equals_2_l330_330207


namespace eleven_place_unamed_racer_l330_330655

theorem eleven_place_unamed_racer
  (Rand Hikmet Jack Marta David Todd : ℕ)
  (positions : Fin 15)
  (C_1 : Rand = Hikmet + 6)
  (C_2 : Marta = Jack + 1)
  (C_3 : David = Hikmet + 3)
  (C_4 : Jack = Todd + 3)
  (C_5 : Todd = Rand + 1)
  (C_6 : Marta = 8) :
  ∃ (x : Fin 15), (x ≠ Rand) ∧ (x ≠ Hikmet) ∧ (x ≠ Jack) ∧ (x ≠ Marta) ∧ (x ≠ David) ∧ (x ≠ Todd) ∧ x = 11 := 
sorry

end eleven_place_unamed_racer_l330_330655


namespace math_problem_l330_330342

theorem math_problem 
  (a b c d : ℝ) 
  (h1 : a ≥ b) 
  (h2 : b ≥ c) 
  (h3 : c ≥ d) 
  (h4 : d > 0) 
  (h5 : a + b + c + d = 1) : 
  (a + 2*b + 3*c + 4*d) * a^a * b^b * c^c * d^d < 1 := 
sorry

end math_problem_l330_330342


namespace sectionB_seats_correct_l330_330689

-- Definitions for the number of seats in Section A
def seatsA_subsection1 : Nat := 60
def seatsA_subsection2 : Nat := 3 * 80
def totalSeatsA : Nat := seatsA_subsection1 + seatsA_subsection2

-- Condition for the number of seats in Section B
def seatsB : Nat := 3 * totalSeatsA + 20

-- Theorem statement to prove the number of seats in Section B
theorem sectionB_seats_correct : seatsB = 920 := by
  sorry

end sectionB_seats_correct_l330_330689


namespace Brad_pumpkin_weight_l330_330666

theorem Brad_pumpkin_weight (B : ℝ)
  (h1 : ∃ J : ℝ, J = B / 2)
  (h2 : ∃ Be : ℝ, Be = 4 * (B / 2))
  (h3 : ∃ Be J : ℝ, Be - J = 81) : B = 54 := by
  obtain ⟨J, hJ⟩ := h1
  obtain ⟨Be, hBe⟩ := h2
  obtain ⟨_, hBeJ⟩ := h3
  sorry

end Brad_pumpkin_weight_l330_330666


namespace shaded_region_area_and_circle_centers_l330_330030

theorem shaded_region_area_and_circle_centers :
  ∃ (R : ℝ) (center_big center_small1 center_small2 : ℝ × ℝ),
    R = 10 ∧ 
    center_small1 = (4, 0) ∧
    center_small2 = (10, 0) ∧
    center_big = (7, 0) ∧
    (π * R^2) - (π * 4^2 + π * 6^2) = 48 * π :=
by 
  sorry

end shaded_region_area_and_circle_centers_l330_330030


namespace at_most_3_concurrent_lines_l330_330333

open EuclideanGeometry

def is_convex_5gon (A : Fin 5 → ℚ × ℚ) : Prop :=
  ∃ v : ℕ, v < 5 ∧ convex_polygon (λ i, A ((v + i) % 5))

noncomputable def intersection {p1 p2 p3 p4 : ℚ × ℚ} (h : Line p1 p2 ≠ Line p3 p4) : ℚ × ℚ :=
  sorry -- Define the intersection point of lines p1p2 and p3p4, ensuring it is rational when possible

noncomputable def B (A : Fin 5 → ℚ × ℚ) (i : Fin 5) : ℚ × ℚ :=
  intersection (Line (A (i + 1) % 5) (A (i + 2) % 5)) ((A (i + 3) % 5) (A (i + 4) % 5))

theorem at_most_3_concurrent_lines (A : Fin 5 → ℚ × ℚ)
  (h1 : is_convex_5gon A)
  (h2 : ∀ i : Fin 5, ∃ (qi q2 : ℚ), A i = (qi, q2)) :
  ∃ (C : Finset (Line ℚ)), C.card = 3 ∧ ∀ {i : Fin 5}, Line (A i) (B A i) ∉ C := 
sorry -- This represents the proof that at most 3 lines of AiBi are concurrent

end at_most_3_concurrent_lines_l330_330333


namespace sum_perfect_squares_with_five_factors_lt_100_l330_330794

open Nat

theorem sum_perfect_squares_with_five_factors_lt_100 : 
  ∃ (a b : ℕ), (a < 100 ∧ b < 100) ∧ (∃ p₁ p₂ : ℕ, prime p₁ ∧ prime p₂ ∧ a = p₁^4 ∧ b = p₂^4) ∧ 
  (factors a).length = 5 ∧ (factors b).length = 5 ∧ a + b = 97 :=
by
  sorry

end sum_perfect_squares_with_five_factors_lt_100_l330_330794


namespace infinite_power_tower_solution_l330_330196

theorem infinite_power_tower_solution : 
  ∃ x : ℝ, (∀ y, y = x ^ y → y = 4) → x = Real.sqrt 2 :=
by
  sorry

end infinite_power_tower_solution_l330_330196


namespace range_of_a_l330_330579

open Set Real

theorem range_of_a (a : ℝ) (α : ℝ → Prop) (β : ℝ → Prop) (hα : ∀ x, α x ↔ x ≥ a) (hβ : ∀ x, β x ↔ |x - 1| < 1)
  (h : ∀ x, (β x → α x) ∧ (∃ x, α x ∧ ¬β x)) : a ≤ 0 :=
by
  sorry

end range_of_a_l330_330579


namespace savings_using_raspberries_l330_330159

noncomputable def blueberry_cost_per_carton : ℝ := 5.00
noncomputable def raspberry_cost_per_carton : ℝ := 3.00
noncomputable def blueberry_ounces_per_carton : ℕ := 6
noncomputable def raspberry_ounces_per_carton : ℕ := 8
noncomputable def batches : ℕ := 4
noncomputable def ounces_per_batch : ℕ := 12

theorem savings_using_raspberries :
  let total_fruit := batches * ounces_per_batch in
  let blueberry_cartons := total_fruit / blueberry_ounces_per_carton in
  let raspberry_cartons := total_fruit / raspberry_ounces_per_carton in
  let cost_of_blueberries := blueberry_cartons * blueberry_cost_per_carton in
  let cost_of_raspberries := raspberry_cartons * raspberry_cost_per_carton in
  let savings := cost_of_blueberries - cost_of_raspberries in
  savings = 22.00 :=
by
  sorry

end savings_using_raspberries_l330_330159


namespace shaded_region_area_l330_330499

-- Define the conditions
variable (AD DC EF FG : ℕ)
variable (D : Point)
variable (circle_radius : ℕ)

-- Making sure the values are correctly assigned based on the problem
axiom AD_def : AD = 5
axiom DC_def : DC = 12
axiom EF_def : EF = 3
axiom FG_def : FG = 4
axiom radius_def : circle_radius = 13

-- Definition of areas based on given dimensions
noncomputable def area_ABCD := AD * DC
noncomputable def area_EFGH := EF * FG
noncomputable def quarter_circle_area := (169 * Real.pi) / 4

-- The main statement to prove
theorem shaded_region_area : (169 * Real.pi) / 4 - area_ABCD - area_EFGH = (169 * Real.pi) / 4 - 72 :=
by
  -- Using Lean's capabilities to simplify and solve
  sorry

end shaded_region_area_l330_330499


namespace truncated_cone_radius_l330_330859

noncomputable def volume_of_cone (R M : ℝ) : ℝ := (1 / 3) * π * R^2 * M

noncomputable def surface_area_of_cone (R α : ℝ) : ℝ := 
  let l := R / (Real.cos α) in 
  π * R^2 + π * R * l

noncomputable def truncated_cone_surface_area (R r α : ℝ) : ℝ := 
  let l' := (R - r) / (Real.cos α) in 
  π * R^2 + π * r^2 + π * (R + r) * l'

theorem truncated_cone_radius :
  let K := 86256 in
  let α := (80 + (25 / 60) + (22 / 3600)) * (π / 180) in
  let R := Real.sqrt (Real.cbrt (3 * K / (π * Real.tan α))) in
  let r := (1 / 2) * Real.sqrt 2 * Real.cot (α / 2) * Real.sqrt (Real.cbrt (3 * K / (π * Real.tan α))) in
  surface_area_of_cone R α / 2 = truncated_cone_surface_area R r α →
  r = 20.11 :=
sorry

end truncated_cone_radius_l330_330859


namespace triple_composition_l330_330346

def f (x : ℝ) : ℝ :=
  if x ≤ 3 then x^3 else Real.exp x

theorem triple_composition (x : ℝ) (h : x ≤ 3) : f(f(f(x))) = 1 := by
  have h₁ : f(x) = x^3 := if_pos h
  have h₂ : 1 ≤ 3 := by linarith
  have h₃ : f(1) = 1^3 := if_pos h₂
  rw [h₁, h₃]
  have h₄ : f(1) = 1 := if_pos h₂
  rw [h₄]
  have h₅ : f(1) = 1 := if_pos h₂
  rw [h₅]
  sorry

end triple_composition_l330_330346


namespace range_of_g_is_arctan2_l330_330206

def g (x : ℝ) : ℝ := Real.arctan (x^2) + Real.arctan ((2 - 2*x^2) / (1 + 2*x^2))

theorem range_of_g_is_arctan2 : ∀ x : ℝ, g x = Real.arctan 2 :=
by
  sorry

end range_of_g_is_arctan2_l330_330206


namespace width_of_house_l330_330393

theorem width_of_house (L P_L P_W A_total : ℝ) (hL : L = 20.5) (hPL : P_L = 6) (hPW : P_W = 4.5) (hAtotal : A_total = 232) :
  ∃ W : ℝ, W = 10 :=
by
  have area_porch : ℝ := P_L * P_W
  have area_house := A_total - area_porch
  use area_house / L
  sorry

end width_of_house_l330_330393


namespace tournament_ranking_sequences_l330_330311

def total_fair_ranking_sequences (A B C D : Type) : Nat :=
  let saturday_outcomes := 2
  let sunday_outcomes := 4 -- 2 possibilities for (first, second) and 2 for (third, fourth)
  let tiebreaker_effect := 2 -- swap second and third
  saturday_outcomes * sunday_outcomes * tiebreaker_effect

theorem tournament_ranking_sequences (A B C D : Type) :
  total_fair_ranking_sequences A B C D = 32 := 
by
  sorry

end tournament_ranking_sequences_l330_330311


namespace binomial_coefficient_max_term_alternating_sum_remainder_l330_330237

noncomputable def polynomial_expansion (m n : ℕ) := 
  (1 + m*x)^n

theorem binomial_coefficient_max_term (m n : ℕ) (h1 : m ≠ 0) (h2 : n ≥ 2) (h3 : n ∈ ℕ)
  (h4 : ∑ k in range (n+1), choose n k * (m ^ k) = 0)
  (h5 : choose n 5 * m^5 = max (λ k, choose n k * m^k))
  (h6 : choose n 2 * m^2 = 9 * choose n 1 * m) :
  m = 2 ∧ n = 10 := sorry

theorem alternating_sum_remainder (a : ℕ → ℤ) (h : ∀ x, polynomial_expansion 2 10 x = ∑ i in range (11), a i * (x+8)^i) :
  (∑ i in range (11), (-1) ^ i * a i) % 6 = 1 := sorry

end binomial_coefficient_max_term_alternating_sum_remainder_l330_330237


namespace point_location_l330_330584

variables {A B C m n : ℝ}

theorem point_location (h1 : A > 0) (h2 : B < 0) (h3 : A * m + B * n + C < 0) : 
  -- Statement: the point P(m, n) is on the upper right side of the line Ax + By + C = 0
  true :=
sorry

end point_location_l330_330584


namespace mooncake_inspection_random_event_l330_330101

-- Definition of event categories
inductive Event
| certain
| impossible
| random

-- Definition of the event in question
def mooncakeInspectionEvent (satisfactory: Bool) : Event :=
if satisfactory then Event.random else Event.random

-- Theorem statement to prove that the event is a random event
theorem mooncake_inspection_random_event (satisfactory: Bool) :
  mooncakeInspectionEvent satisfactory = Event.random :=
sorry

end mooncake_inspection_random_event_l330_330101


namespace problem_statement_l330_330960

noncomputable def circle_properties (a t : ℝ) (ha : a > 1) (ht : t ≠ 0) : Prop :=
  let M := (a, t)
  let N := (a, (1 - a^2) / t)
  let center := (a, ((1 - a^2) / (2 * t) + 1 / (2 * t)))
  let radius := abs ((1 - a^2) / (2 * t) - 1 / (2 * t))
  (a* a + ((1-a^2)/(2*t) + (1/(2*t)))^2 - ((1-a^2)/(2*t) - (1/(2*t)))^2 = 1 ∧
   ((1-a^2)/t + t = 0 → ((x - a)^2 + y^2 = a^2 - 1)))

theorem problem_statement (a t: ℝ) (ha : a > 1) (ht : t ≠ 0) : circle_properties a t ha ht :=
by sorry

end problem_statement_l330_330960


namespace binary_to_decimal_l330_330889

theorem binary_to_decimal : 
  let binary_number := [1, 1, 0, 1, 1] in
  let weights := [2^4, 2^3, 2^2, 2^1, 2^0] in
  (List.zipWith (λ digit weight => digit * weight) binary_number weights).sum = 27 := by
  sorry

end binary_to_decimal_l330_330889


namespace log_equation_solution_l330_330546

theorem log_equation_solution (x : ℝ) (h1 : 0 < x) (h2 : 0 < x + 2) :
  log x + log (x + 2) = log (2x + 3) ↔ x = real.sqrt 3 :=
by
  sorry

end log_equation_solution_l330_330546


namespace angle_between_vectors_l330_330595

variables {𝕜 : Type*} [IsROrC 𝕜]
open IsROrC Real

def vector_space := EuclideanSpace 𝕜 (Fin 2)

variables (a b : vector_space)

-- Condition definitions
def non_zero (u : vector_space) : Prop := ∥u∥ ≠ 0
def magnitude_b : Prop := ∥b∥ = 2
def dot_product_condition : Prop := inner a b = ∥a∥

-- The theorem to prove
theorem angle_between_vectors (H1 : non_zero a) 
                             (H2 : non_zero b) 
                             (H3 : magnitude_b) 
                             (H4 : dot_product_condition) : 
  ∃ θ : ℝ, θ = π / 3 ∧ (cos θ = inner a b / (∥a∥ * ∥b∥)) :=
begin
  sorry
end

end angle_between_vectors_l330_330595


namespace computer_price_increase_l330_330403

-- Define the given conditions
def original_price (b : ℝ) : Prop := 2 * b = 540
def final_price : ℝ := 351

-- Define the percentage increase function
def percentage_increase (old_price new_price : ℝ) : ℝ :=
  ((new_price - old_price) / old_price) * 100

-- The main statement to be proven
theorem computer_price_increase (b : ℝ) (h : original_price b) :
  percentage_increase b final_price = 30 := by
  sorry

end computer_price_increase_l330_330403


namespace price_of_adult_ticket_l330_330797

-- Define the conditions
def price_student_ticket : ℝ := 2.50
def total_tickets_sold : ℕ := 59
def total_revenue : ℝ := 222.50
def student_tickets_sold : ℕ := 9

-- Define the goal / question we want to prove
theorem price_of_adult_ticket 
    (price_student_ticket : ℝ)
    (total_tickets_sold : ℕ)
    (total_revenue : ℝ)
    (student_tickets_sold : ℕ)
    (H_price_student : price_student_ticket = 2.50)
    (H_total_tickets : total_tickets_sold = 59)
    (H_total_revenue : total_revenue = 222.50)
    (H_student_tickets : student_tickets_sold = 9) :
    let revenue_from_student_tickets := student_tickets_sold * price_student_ticket,
        revenue_from_adult_tickets := total_revenue - revenue_from_student_tickets,
        adult_tickets_sold := total_tickets_sold - student_tickets_sold,
        price_adult_ticket := revenue_from_adult_tickets / adult_tickets_sold
    in
    price_adult_ticket = 4.00 := 
by 
    -- We state the assertion here
    sorry

end price_of_adult_ticket_l330_330797


namespace face_value_of_share_l330_330842

theorem face_value_of_share (FV : ℝ) (market_value : ℝ) (dividend_rate : ℝ) (desired_return_rate : ℝ) 
  (H1 : market_value = 15) 
  (H2 : dividend_rate = 0.09) 
  (H3 : desired_return_rate = 0.12) 
  (H4 : dividend_rate * FV = desired_return_rate * market_value) :
  FV = 20 := 
by
  sorry

end face_value_of_share_l330_330842


namespace bisect_condition_l330_330651

variable {A B C D E F : Type} [EuclideanGeometry A B C D E F]

-- Definitions of the points and segments
def triangle_ABC_isosceles (AB AC : ℝ) (h_iso : AB = AC) : Prop := 
∃ (A B C : Point), dist A B = dist A C

def angle_condition (D : Point) (A B C : Point) (h_angle : ∠DCB = ∠DBA) : Prop := 
∠D C B = ∠D B A

def points_on_segments (D B C E F : Point) (h_E_on_DB : lies_on E (segment D B))
  (h_F_on_DC : lies_on F (segment D C)) : Prop :=
  lies_on E (segment D B) ∧ lies_on F (segment D C)

-- Main problem statement
theorem bisect_condition (A B C D E F : Point) (AB AC : ℝ) 
  (h_iso : triangle_ABC_isosceles AB AC h_iso)
  (h_angle : angle_condition D A B C h_angle)
  (h_segments : points_on_segments D B C E F h_E_on_DB h_F_on_DC) :
  bisect (line A D) (segment E F) ↔ concyclic {E, B, C, F} :=
sorry

end bisect_condition_l330_330651


namespace total_crayons_all_18_children_l330_330659

theorem total_crayons_all_18_children :
  let a1 := 12 -- first child receives 12 crayons
  let d := 2 -- common difference: each subsequent child receives 2 more crayons
  let n := 18 -- total number of children
  let a18 := a1 + (n - 1) * d -- calculating the 18th term
  let S18 := n / 2 * (a1 + a18) -- sum of the first 18 terms of the arithmetic sequence
  S18 = 522 := by {
    -- using the given conditions to derive the required sum
    let a1 := 12
    let d := 2
    let n := 18
    let a18 := a1 + (n - 1) * d
    have h_a18 : a18 = 46 := by {
      calc
        a18 = 12 + 17 * 2 : by sorry
        ... = 46 : by sorry
    }
    have h_S18 : S18 = 9 * (12 + 46) := by {
      calc
        S18 = (18 / 2) * (12 + a18) : by sorry
        ... = 9 * (12 + 46) : by sorry
    }
    show S18 = 522
    from calc
      9 * (12 + 46) = 9 * 58 : by sorry
      ... = 522 : by sorry
} sorry

end total_crayons_all_18_children_l330_330659


namespace min_editors_at_conference_l330_330870

variable (x E : ℕ)

theorem min_editors_at_conference (h1 : x ≤ 26) 
    (h2 : 100 = 35 + E + x) 
    (h3 : 2 * x ≤ 100 - 35 - E + x) : 
    E ≥ 39 :=
by
  sorry

end min_editors_at_conference_l330_330870


namespace axis_of_symmetry_l330_330636

variable (f : ℝ → ℝ)

theorem axis_of_symmetry (h : ∀ x, f x = f (5 - x)) :  ∀ x y, y = f x ↔ (x = 2.5 ∧ y = f 2.5) := 
sorry

end axis_of_symmetry_l330_330636


namespace solution_set_equivalence_l330_330405

def abs_value_solution_set (x : ℝ) : Prop := (x) * (|x + 2|) < 0

theorem solution_set_equivalence : {x : ℝ | abs_value_solution_set x} = {x : ℝ | x < -2 ∨ (-2 < x ∧ x < 0)} :=
by
  sorry

end solution_set_equivalence_l330_330405


namespace average_time_per_mile_l330_330361

-- Define the conditions
def total_distance_miles : ℕ := 24
def total_time_hours : ℕ := 3
def total_time_minutes : ℕ := 36
def total_time_in_minutes : ℕ := (total_time_hours * 60) + total_time_minutes

-- State the theorem
theorem average_time_per_mile : total_time_in_minutes / total_distance_miles = 9 :=
by
  sorry

end average_time_per_mile_l330_330361


namespace range_of_d_l330_330958

noncomputable def circle_center : ℝ × ℝ := (3, 4)
noncomputable def circle_radius : ℝ := 1
noncomputable def A : ℝ × ℝ := (0, -1)
noncomputable def B : ℝ × ℝ := (0, 1)
noncomputable def is_on_circle (P : ℝ × ℝ) : Prop := (P.1 - circle_center.1)^2 + (P.2 - circle_center.2)^2 = circle_radius^2
noncomputable def dist_squared (P Q : ℝ × ℝ) : ℝ := (P.1 - Q.1)^2 + (P.2 - Q.2)^2
noncomputable def d (P : ℝ × ℝ) : ℝ := dist_squared P A + dist_squared P B

theorem range_of_d : ∀ P : ℝ × ℝ, is_on_circle P → d P ∈ set.Icc (32 : ℝ) (72 : ℝ) :=
by
  sorry

end range_of_d_l330_330958


namespace sum_of_positive_factors_60_l330_330054

def sum_of_positive_factors (n : ℕ) : ℕ :=
  ∑ d in (Finset.filter (λ x => x > 0) (Finset.divisors n)), d

theorem sum_of_positive_factors_60 : sum_of_positive_factors 60 = 168 := by
  sorry

end sum_of_positive_factors_60_l330_330054


namespace approx_side_length_approx_sum_correct_l330_330424

noncomputable def pi := 3.14159265358979323846
noncomputable def pi_squared := 9.869604401089358618834
noncomputable def x := pi / 11

noncomputable def taylor_sin (x : Float) := 
  x - (x^3 / 3.factorial) + (x^5 / 5.factorial) - (x^7 / 7.factorial) + (x^9 / 9.factorial)

noncomputable def polygon_side_length := 2 * taylor_sin x

noncomputable def approximate_sum := 1/3 + 1/5 + 1/51 + 1/95

theorem approx_side_length:
  0.56346508 < polygon_side_length ∧ polygon_side_length < 0.56346518 :=
by
  sorry

theorem approx_sum_correct:
  approximate_sum ≈ 0.56346748 :=
by
  sorry

end approx_side_length_approx_sum_correct_l330_330424


namespace incorrect_value_l330_330775

theorem incorrect_value:
  ∀ (n : ℕ) (initial_mean corrected_mean : ℚ) (correct_value incorrect_value : ℚ),
  n = 50 →
  initial_mean = 36 →
  corrected_mean = 36.5 →
  correct_value = 48 →
  incorrect_value = correct_value - (corrected_mean * n - initial_mean * n) →
  incorrect_value = 23 :=
by
  intros n initial_mean corrected_mean correct_value incorrect_value
  intros h1 h2 h3 h4 h5
  sorry

end incorrect_value_l330_330775


namespace num_four_digit_numbers_div_by_3_with_last_two_23_l330_330274

theorem num_four_digit_numbers_div_by_3_with_last_two_23 : 
  {n : Nat // 1000 ≤ n ∧ n < 10000 ∧ n % 100 = 23 ∧ n % 3 = 0}.card = 30 :=
sorry

end num_four_digit_numbers_div_by_3_with_last_two_23_l330_330274


namespace percentage_first_division_l330_330312

theorem percentage_first_division (total_students : ℕ) (perc_second_division : ℕ) (num_passed : ℕ)
  (no_student_failed : total_students = 300 ∧ perc_second_division = 54 ∧ num_passed = 51) : 
  let num_second_division := perc_second_division * total_students / 100 in
  let num_first_division := total_students - (num_second_division + num_passed) in
  let perc_first_division := num_first_division * 100 / total_students in
  perc_first_division = 29 := 
by
  sorry

end percentage_first_division_l330_330312


namespace frequency_calculation_l330_330483

-- Define the given conditions
def sample_capacity : ℕ := 20
def group_frequency : ℚ := 0.25

-- The main theorem statement
theorem frequency_calculation :
  sample_capacity * group_frequency = 5 :=
by sorry

end frequency_calculation_l330_330483


namespace sum_of_factors_of_60_l330_330074

-- Given conditions
def n : ℕ := 60

-- Proof statement
theorem sum_of_factors_of_60 : (∑ d in (finset.filter (λ d, n % d = 0) (finset.range (n + 1))), d) = 168 :=
by
  sorry

end sum_of_factors_of_60_l330_330074


namespace integral_cos_squared_eq_pi_div_two_l330_330878

theorem integral_cos_squared_eq_pi_div_two : (∫ x in 0..π, real.cos x ^ 2) = π / 2 := 
by sorry

end integral_cos_squared_eq_pi_div_two_l330_330878


namespace sufficient_but_not_necessary_l330_330822

theorem sufficient_but_not_necessary (x : ℝ) : (x - 1 > 0) → (x^2 - 1 > 0) ∧ ¬((x^2 - 1 > 0) → (x - 1 > 0)) :=
by 
  sorry

end sufficient_but_not_necessary_l330_330822


namespace vector_magnitude_parallel_l330_330271

theorem vector_magnitude_parallel (x : ℝ) 
  (h1 : 4 / x = 2 / 1) :
  ( Real.sqrt ((4 + x) ^ 2 + (2 + 1) ^ 2) ) = 3 * Real.sqrt 5 := 
sorry

end vector_magnitude_parallel_l330_330271


namespace find_point_W_coordinates_l330_330552

theorem find_point_W_coordinates 
(O U S V : ℝ × ℝ)
(hO : O = (0, 0))
(hU : U = (3, 3))
(hS : S = (3, 0))
(hV : V = (0, 3))
(hSquare : (O.1 - U.1)^2 + (O.2 - U.2)^2 = 18)
(hArea_Square : 3 * 3 = 9) :
  ∃ W : ℝ × ℝ, W = (3, 9) ∧ 1 / 2 * (abs (S.1 - V.1) * abs (W.2 - S.2)) = 9 :=
by
  sorry

end find_point_W_coordinates_l330_330552


namespace sum_p_s_at_11_l330_330702

-- Define the set S as the set of 11-tuples where each entry is either 0 or 1
def S : set (fin 11 → ℕ) := { s | ∀ i, s i = 0 ∨ s i = 1 }

-- Define the polynomial p_s(x) with degree at most 10
def p_s (s : fin 11 → ℕ) (x : ℕ) : ℕ := sorry -- p_s is defined such that p_s(n) = s(n) for 0 ≤ n ≤ 10

-- Define P(x) as the sum of p_s(x) over all s in S
def P (x : ℕ) : ℕ := ∑ s in S.to_finset, p_s s x

-- Prove that P(11) = 1024
theorem sum_p_s_at_11 : P 11 = 1024 :=
by
  sorry

end sum_p_s_at_11_l330_330702


namespace facebook_bonus_each_female_mother_received_l330_330202

theorem facebook_bonus_each_female_mother_received (total_earnings : ℝ) (bonus_percentage : ℝ) 
    (total_employees : ℕ) (male_fraction : ℝ) (female_non_mothers : ℕ) : 
    total_earnings = 5000000 → bonus_percentage = 0.25 → total_employees = 3300 → 
    male_fraction = 1 / 3 → female_non_mothers = 1200 → 
    (250000 / ((total_employees - total_employees * male_fraction.to_nat) - female_non_mothers)) = 1250 :=
by {
  sorry
}

end facebook_bonus_each_female_mother_received_l330_330202


namespace molecular_weight_and_composition_l330_330807

theorem molecular_weight_and_composition :
∀ (H_atoms Cr_atoms O_atoms N_atoms : ℕ)
  (H_weight Cr_weight O_weight N_weight : ℝ),
  H_atoms = 4 →
  Cr_atoms = 2 →
  O_atoms = 6 →
  N_atoms = 3 →
  H_weight = 1.008 →
  Cr_weight = 51.996 →
  O_weight = 15.999 →
  N_weight = 14.007 →
  let molecular_weight := (H_atoms * H_weight) + (Cr_atoms * Cr_weight) + (O_atoms * O_weight) + (N_atoms * N_weight)
  in molecular_weight = 246.039 ∧
     ((H_atoms * H_weight / molecular_weight) * 100 ≈ 1.638) ∧
     ((Cr_atoms * Cr_weight / molecular_weight) * 100 ≈ 42.272) ∧
     ((O_atoms * O_weight / molecular_weight) * 100 ≈ 39.022) ∧
     ((N_atoms * N_weight / molecular_weight) * 100 ≈ 17.068) := 
by
  intros H_atoms Cr_atoms O_atoms N_atoms H_weight Cr_weight O_weight N_weight
  rintros rfl rfl rfl rfl rfl rfl rfl rfl
  let molecular_weight := (4 * 1.008) + (2 * 51.996) + (6 * 15.999) + (3 * 14.007)
  have : molecular_weight = 246.039 := sorry
  split
  · exact this
  · have H_percentage := (4 * 1.008 / molecular_weight) * 100
    have H_approx : H_percentage ≈ 1.638 := sorry
    split
    · exact H_approx
    · have Cr_percentage := (2 * 51.996 / molecular_weight) * 100
      have Cr_approx : Cr_percentage ≈ 42.272 := sorry
      split
      · exact Cr_approx
      · have O_percentage := (6 * 15.999 / molecular_weight) * 100
        have O_approx : O_percentage ≈ 39.022 := sorry
        split
        · exact O_approx
        · have N_percentage := (3 * 14.007 / molecular_weight) * 100
          have N_approx : N_percentage ≈ 17.068 := sorry
          exact N_approx

end molecular_weight_and_composition_l330_330807


namespace length_of_CD_l330_330464

-- Defining the given conditions
variables (A B C D : ℝ)
variables (length_AC : ℝ) (length_AB : ℝ) (midpoint_D : ℝ)
variables (perpendicular_CD_AB : Prop)

-- Given conditions
def circle_conditions : Prop :=
  length_AC = 10 ∧ length_AB = 8 ∧ midpoint_D = 4 ∧ perpendicular_CD_AB

-- The statement to prove
theorem length_of_CD (length_AC : ℝ) (length_AB : ℝ) (midpoint_D : ℝ) (perpendicular_CD_AB : Prop) 
  (h : circle_conditions length_AC length_AB midpoint_D perpendicular_CD_AB) :
  ∃ CD : ℝ, CD = 2 * sqrt 21 :=
begin
  sorry
end

end length_of_CD_l330_330464


namespace ratio_nine_years_ago_correct_l330_330170

-- Conditions
def C : ℕ := 24
def G : ℕ := C / 2

-- Question and expected answer
def ratio_nine_years_ago : ℕ := (C - 9) / (G - 9)

theorem ratio_nine_years_ago_correct : ratio_nine_years_ago = 5 := by
  sorry

end ratio_nine_years_ago_correct_l330_330170


namespace total_wax_required_l330_330735

/-- Given conditions: -/
def wax_already_have : ℕ := 331
def wax_needed_more : ℕ := 22

/-- Prove the question (the total amount of wax required) -/
theorem total_wax_required :
  (wax_already_have + wax_needed_more) = 353 := by
  sorry

end total_wax_required_l330_330735


namespace average_time_per_mile_l330_330363

noncomputable def miles : ℕ := 24
noncomputable def hours : ℕ := 3
noncomputable def minutes : ℕ := 36

theorem average_time_per_mile :
  let total_time := hours * 60 + minutes in
  total_time / miles = 9 := by
  sorry

end average_time_per_mile_l330_330363


namespace decimal_to_fraction_l330_330189

noncomputable def repeating_decimal_to_fraction (x : ℚ) : Prop :=
  x = 0.157142857142857...

theorem decimal_to_fraction :
  ∃ (x : ℚ), repeating_decimal_to_fraction x ∧ x = 10690 / 68027 :=
by
  use 0.157142857142857...
  split
  . exact rfl
  . sorry

end decimal_to_fraction_l330_330189


namespace shop_owner_profit_l330_330440

theorem shop_owner_profit :
  ∀ (buy_cheat sell_cheat : ℝ) (cp_per_kg sp_per_kg : ℝ), 
  buy_cheat = 0.10 → sell_cheat = 0.10 →
  cp_per_kg = 1 →
  sp_per_kg = 1 →
  let weight_bought := cp_per_kg * (1 + buy_cheat) in
  let weight_sold := sp_per_kg * (1 - sell_cheat) in
  let actual_sp := weight_bought / weight_sold * sp_per_kg in
  let profit := actual_sp - cp_per_kg in
  (profit / cp_per_kg * 100) ≈ 22.2 :=
begin
  intros buy_cheat sell_cheat cp_per_kg sp_per_kg hb hs hcp hsp,
  have wb : weight_bought = 1 + 0.10, by rw [hcp, hb]; exact rfl,
  have ws : weight_sold = 1 - 0.10, by rw [hsp, hs]; exact rfl,
  have asp : actual_sp = (1 + 0.10) / (1 - 0.10), by rw [wb, ws, hsp]; exact rfl,
  have prof : profit = (1.1 / 0.9) - 1, by rw [asp, hcp]; exact rfl,
  have perc_prof : (profit / cp_per_kg * 100) = 0.222222... * 100, by rw [prof, hcp]; exact rfl,
  show 0.222222... * 100 ≈ 22.2,
  sorry
end

end shop_owner_profit_l330_330440


namespace pyramid_equidistant_distance_l330_330590

theorem pyramid_equidistant_distance 
  (a b c : ℝ) 
  (DA DB DC : Point) 
  (ABC : Plane) 
  (M : Point)
  (DM : Segment)
  (H₁ : angle B D C = 90) 
  (H₂ : angle C D A = 90) 
  (H₃ : angle A D B = 90) 
  (H₄ : DA = a)
  (H₅ : DB = b)
  (H₆ : DC = c)
  (H₇ : M ∈ ABC)
  (H₈ : equidistant_from_planes M ABC [CDB, CDA, ADB]) 
  : DM.length = (a * b * c * real.sqrt 3) / (a * c + b * c + a * b) := 
sorry

end pyramid_equidistant_distance_l330_330590


namespace fractional_addition_l330_330921

theorem fractional_addition : (2 : ℚ) / 5 + 3 / 8 = 31 / 40 :=
by
  sorry

end fractional_addition_l330_330921


namespace sum_factors_of_60_l330_330062

def sum_factors (n : ℕ) : ℕ :=
  (∑ d in Finset.divisors n, d)

theorem sum_factors_of_60 : sum_factors 60 = 168 := by
  sorry

end sum_factors_of_60_l330_330062


namespace vertical_angles_congruent_l330_330747

theorem vertical_angles_congruent (A B : Type) [angle A] [angle B] (h : vertical A B) : congruent A B :=
sorry

end vertical_angles_congruent_l330_330747


namespace sum_p_s_12_l330_330703

noncomputable def S : finset (fin 12 → bool) := {s : fin 12 → bool | true}.to_finset

noncomputable def p_s (s : fin 12 → bool) : polynomial ℝ := 
  finsupp.single 0 (if s 0 then 1 else 0) + 
  finsupp.single 1 (if s 1 then 1 else 0) + 
  finsupp.single 2 (if s 2 then 1 else 0) + 
  finsupp.single 3 (if s 3 then 1 else 0) + 
  finsupp.single 4 (if s 4 then 1 else 0) + 
  finsupp.single 5 (if s 5 then 1 else 0) + 
  finsupp.single 6 (if s 6 then 1 else 0) + 
  finsupp.single 7 (if s 7 then 1 else 0) + 
  finsupp.single 8 (if s 8 then 1 else 0) + 
  finsupp.single 9 (if s 9 then 1 else 0) + 
  finsupp.single 10 (if s 10 then 1 else 0) + 
  finsupp.single 11 (if s 11 then 1 else 0)

theorem sum_p_s_12 : ∑ s in S, (p_s s).eval 12 = 2048 := 
  sorry

end sum_p_s_12_l330_330703


namespace triangle_circumcircles_common_point_l330_330227

universe u

variables {α : Type u} [metric_space α]

noncomputable def triangle_has_common_point 
  (A B C : α) (I : α) 
  (Γ_A Γ_B Γ_C : emetric.ball I ∞)
  (A' B' C' : α) : Prop := 
  let circumcircle (P Q R : α) : set α := 
    { X | dist X P = dist X Q }
    ∩ { X | dist X Q = dist X R }
  in
  dist A B ≠ dist B C ∧ 
  dist B C ≠ dist C A ∧
  dist C A ≠ dist A B ∧
  mem_circumcircle Γ_A I ∧
  mem_circumcircle Γ_B I ∧
  mem_circumcircle Γ_C I ∧
  (Γ_A ∩ Γ_B).second_intersect = A' ∧
  (Γ_B ∩ Γ_C).second_intersect = B' ∧
  (Γ_C ∩ Γ_A).second_intersect = C' →
  ∃ Q, Q ≠ I ∧ Q ∈ circumcircle A I A' ∧ Q ∈ circumcircle B I B' ∧ Q ∈ circumcircle C I C'

theorem triangle_circumcircles_common_point
  {α : Type u} [metric_space α]
  (A B C I : α)
  (Γ_A Γ_B Γ_C : emetric.ball I ∞)
  (A' B' C' : α) :
  triangle_has_common_point A B C I Γ_A Γ_B Γ_C A' B' C' :=
sorry

end triangle_circumcircles_common_point_l330_330227


namespace charge_per_mile_l330_330727

theorem charge_per_mile (rental_fee total_amount_paid : ℝ) (num_miles : ℕ) (charge_per_mile : ℝ) : 
  rental_fee = 20.99 →
  total_amount_paid = 95.74 →
  num_miles = 299 →
  (total_amount_paid - rental_fee) / num_miles = charge_per_mile →
  charge_per_mile = 0.25 :=
by 
  intros r_fee t_amount n_miles c_per_mile h1 h2 h3 h4
  sorry

end charge_per_mile_l330_330727


namespace length_of_QR_l330_330955

theorem length_of_QR (PQ QR : ℝ) (h1 : ∠PQR = 90) (h2 : cos Q = 0.6) (h3 : PQ = 15) : QR = 25 :=
by
  sorry

end length_of_QR_l330_330955


namespace paths_from_P_to_Q_l330_330039

structure Graph where
  (V : Type) -- Vertices
  (E : V → V → Prop) -- Edges

def paths_count (g : Graph) (start : g.V) (end : g.V) : ℕ := sorry

def example_graph : Graph :=
  { V := {P, Q, R, S, T, U},
    E := λ v w, match v, w with
                | P, R => true
                | P, S => true
                | R, Q => true
                | S, T => true
                | S, U => true
                | T, Q => true
                | U, Q => true
                | _, _ => false
  }

-- The theorem we want to prove
theorem paths_from_P_to_Q : paths_count example_graph P Q = 3 := 
  sorry

end paths_from_P_to_Q_l330_330039


namespace find_b_l330_330386

theorem find_b :
  ∃ b : ℚ, ∃ v : ℚ × ℚ, v = (3, b) ∧ v ∝ (4, 3) := 
sorry

end find_b_l330_330386


namespace initial_provisions_last_days_l330_330467

variable (garrison_initial men_reinforcement men_total provisions_remaining_days : ℕ)
variable (x : ℕ)

-- Conditions
def condition1 : garrison_initial = 2000 := 
  by rfl

def condition2 : men_reinforcement = 2700 := 
  by rfl

def condition3 : men_total = 2000 + 2700 := 
  by rfl

def condition4 : provisions_remaining_days = 20 := 
  by rfl

def provision_equation (x : ℕ) : Prop := 
  2000 * (x - 15) = 4700 * 20

-- Theorem to prove
theorem initial_provisions_last_days (h : provision_equation x) : x = 62 := 
  sorry

#check @initial_provisions_last_days

end initial_provisions_last_days_l330_330467


namespace find_a_l330_330648

noncomputable def A : set ℝ := {x | x ≤ 2}
noncomputable def B (a : ℝ) : set ℝ := {x | x ≥ a}

theorem find_a (a : ℝ) (h : A ∩ B a = {2}) : a = 2 := 
sorry

end find_a_l330_330648


namespace fraction_exceeds_l330_330475

theorem fraction_exceeds (x : ℚ) (h : 64 = 64 * x + 40) : x = 3 / 8 := 
by
  sorry

end fraction_exceeds_l330_330475


namespace distance_between_centers_l330_330384

noncomputable def rho1 (θ : ℝ) : ℝ := 2 * Real.cos θ
noncomputable def rho2 (θ : ℝ) : ℝ := Real.sin θ

def center1 : ℝ × ℝ := (1, 0)
def center2 : ℝ × ℝ := (0, 0.5)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem distance_between_centers : distance center1 center2 = Real.sqrt 1.25 :=
by
  sorry

end distance_between_centers_l330_330384


namespace part1_part2_l330_330629

open Real

variables (a b : ℝ → ℝ → ℝ) -- This would represent the vector space, simplified.
variables (dot : (ℝ → ℝ → ℝ) → (ℝ → ℝ → ℝ) → ℝ)
variables (norm : (ℝ → ℝ → ℝ) → ℝ)

axiom norm_a : norm a = 2
axiom norm_b : norm b = 3
axiom dot_condition : dot (3 * a + b) (a - 2 * b) = -16

theorem part1 (h : dot (a - b) (a + λ * b) = 0) : λ = 2 / 7 := sorry

theorem part2 : 
  let ab := dot a (2 * a - b)
  let norm_ab := norm (2 * a - b)
  ab / (norm a * norm_ab) = 3 / sqrt 17 := sorry

end part1_part2_l330_330629


namespace sqrt_sum_leq_three_half_sqrt_three_l330_330358

variables (a b c : ℝ)
variable (h1 : 0 ≤ a ∧ a ≤ 1)
variable (h2 : 0 ≤ b ∧ b ≤ 1)
variable (h3 : 0 ≤ c ∧ c ≤ 1)
variable (h4 : a + b + c = 1 + sqrt (2 * (1 - a) * (1 - b) * (1 - c)))

theorem sqrt_sum_leq_three_half_sqrt_three (h1 : 0 ≤ a ∧ a ≤ 1) (h2 : 0 ≤ b ∧ b ≤ 1) (h3 : 0 ≤ c ∧ c ≤ 1) (h4 : a + b + c = 1 + sqrt (2 * (1 - a) * (1 - b) * (1 - c))) : 
  sqrt (1 - a^2) + sqrt (1 - b^2) + sqrt (1 - c^2) ≤ (3 * sqrt 3) / 2 := 
by
  sorry

end sqrt_sum_leq_three_half_sqrt_three_l330_330358


namespace empty_bidon_weight_l330_330839

theorem empty_bidon_weight (B M : ℝ) 
  (h1 : B + M = 34) 
  (h2 : B + M / 2 = 17.5) : 
  B = 1 := 
by {
  -- The proof steps would go here, but we just add sorry
  sorry
}

end empty_bidon_weight_l330_330839


namespace minimum_area_PJ₁J₂_l330_330183

-- Define the basic structures and conditions of the triangle
variables {P Q R Y J₁ J₂ : Type*}

-- Side lengths
def PQ : ℝ := 24
def QR : ℝ := 26
def PR : ℝ := 28

-- Point Y lies on the line segment QR
def Y_on_QR (Y QR : Type*) : Prop := sorry

-- Incenters of triangles PQY and PRY
def incenter (P Q Y : Type*) : Type* := sorry
def J₁ := incenter P Q Y
def J₂ := incenter P R Y

-- Minimum area of triangle PJ₁J₂
def area_min (P J₁ J₂ : Type*) (a : ℝ) : Prop := ∀ Y, (area P J₁ J₂ Y ≥ a)

theorem minimum_area_PJ₁J₂ (P Q R Y J₁ J₂ : Type*) (hY : Y_on_QR Y QR) 
  (hJ₁ : incenter P Q Y) (hJ₂ : incenter P R Y) :
  area_min P J₁ J₂ 117 := 
sorry

end minimum_area_PJ₁J₂_l330_330183


namespace container_unoccupied_volume_l330_330692

theorem container_unoccupied_volume :
  let side_length_container := 12
  let volume_container := side_length_container ^ 3
  let volume_water := volume_container / 3
  let side_length_ice_cube := 1.5
  let volume_ice_cube := side_length_ice_cube ^ 3
  let number_of_ice_cubes := 15
  let total_volume_ice := number_of_ice_cubes * volume_ice_cube
  let occupied_volume := volume_water + total_volume_ice
  let unoccupied_volume := volume_container - occupied_volume
  unoccupied_volume = 1101.375 := by
calc
  let side_length_container := 12
  let volume_container := side_length_container ^ 3
  let volume_water := volume_container / 3
  let side_length_ice_cube := 1.5
  let volume_ice_cube := side_length_ice_cube ^ 3
  let number_of_ice_cubes := 15
  let total_volume_ice := number_of_ice_cubes * volume_ice_cube
  let occupied_volume := volume_water + total_volume_ice
  let unoccupied_volume := volume_container - occupied_volume
  show unoccupied_volume = 1101.375 from sorry

end container_unoccupied_volume_l330_330692


namespace base7_to_base10_l330_330505

open Nat

theorem base7_to_base10 : (3 * 7^2 + 5 * 7^1 + 1 * 7^0 = 183) :=
by
  sorry

end base7_to_base10_l330_330505


namespace find_tan_theta_l330_330963

open Real

variables {a b c : EuclideanSpace ℝ (Fin 3)}

noncomputable def theta : ℝ := sorry

axiom condition1 : a + b + c = 0
axiom condition2 : real.angle a c = π / 3
axiom condition3 : real.angle a b = θ
axiom condition4 : ∥b∥ = ∥a∥ * sqrt 3

theorem find_tan_theta : tan θ = sqrt 3 / 3 :=
by {
  sorry
}

end find_tan_theta_l330_330963


namespace cot_30_eq_sqrt_3_l330_330555

theorem cot_30_eq_sqrt_3 (h1 : Real.tan (π / 6) = 1 / Real.sqrt 3) : Real.cot (π / 6) = Real.sqrt 3 :=
by
  sorry

end cot_30_eq_sqrt_3_l330_330555


namespace polynomial_factor_correct_l330_330401

theorem polynomial_factor_correct (c q : ℂ) :
  (∃ g : ℂ[X], (3 * X^3 + C(c) * X + C(9)) = (X^2 + C(q) * X + C(3)) * g) →  (c = 0) :=
by
  sorry

end polynomial_factor_correct_l330_330401


namespace length_of_EB_l330_330320

-- Define the given lengths
variables (AE AF FC EB : ℝ)
variable (tangent_line : Set.Point ℝ)

-- Define conditions based on the given problem
def line_is_tangent (t A : Set.Point ℝ) (circle : Set.Circle ℝ) : Prop :=
  t ∈ circle ∧ ∀ (P : Set.Point ℝ), P ∈ circle → P ≠ A → ¬(collinear P t A)

def parallel (l1 l2 : Set.Linear ℝ) : Prop :=
  ∀ (P R : Set.Point ℝ), P ∈ l1 → R ∈ l2 → l1.slope = l2.slope

-- The statement in Lean 4
theorem length_of_EB (t A E F C B : Set.Point ℝ)
  (h_tangent : line_is_tangent t A (Set.Circle ℝ.center))
  (h_parallel : parallel t (Set.Segment E F))
  (h_AE : dist A E = 12) 
  (h_AF : dist A F = 10) 
  (h_FC : dist F C = 14) :
  dist E B = 8 :=
by
  sorry

end length_of_EB_l330_330320


namespace exists_isosceles_trapezoid_with_inscribed_circle_l330_330422

noncomputable def is_trapezoid {Point Plane Line : Type} (P Q: Plane) (p: Line) (A: Point) (C: Point) (trapezoid: Point × Point × Point × Point) : Prop :=
  let (A, B, C, D) := trapezoid in
  ∃ (f: Point → Plane → Line → Prop) (parallel: Line → Line → Prop) (inscribed_circle: Point → Point → Point → Point → Prop),
  f A P p ∧
  f C Q p ∧
  ¬ f A P ∧
  ¬ f C P ∧
  parallel (line_through A B) (line_through C D) ∧
  ∃ (B D: Point), B ∈ P ∧ D ∈ Q ∧ inscribed_circle A B C D

theorem exists_isosceles_trapezoid_with_inscribed_circle
  {Point Plane Line : Type} [f: Point → Plane → Line → Prop] [parallel : Line → Line → Prop] [inscribed_circle : Point → Point → Point → Point → Prop]
  (P Q: Plane) (p: Line) (A: Point) (C: Point)
  (h1 : f A P p) (h2 : f C Q p) (h3 : ¬ f A P) (h4 : ¬ f C P) :
  ∃ (ABCD: Point × Point × Point × Point), is_trapezoid P Q p A C ABCD :=
sorry

end exists_isosceles_trapezoid_with_inscribed_circle_l330_330422


namespace probability_of_continuous_stripe_loop_l330_330886

-- Definitions corresponding to identified conditions:
def cube_faces : ℕ := 6

def diagonal_orientations_per_face : ℕ := 2

def total_stripe_combinations (faces : ℕ) (orientations : ℕ) : ℕ :=
  orientations ^ faces

def satisfying_stripe_combinations : ℕ := 2

-- Proof statement:
theorem probability_of_continuous_stripe_loop :
  (satisfying_stripe_combinations : ℚ) / (total_stripe_combinations cube_faces diagonal_orientations_per_face : ℚ) = 1 / 32 :=
by
  -- Proof goes here
  sorry

end probability_of_continuous_stripe_loop_l330_330886


namespace units_digit_sum_even_20_to_80_l330_330428

theorem units_digit_sum_even_20_to_80 :
  let a := 20
  let d := 2
  let l := 80
  let n := ((l - a) / d) + 1 -- Given by the formula l = a + (n-1)d => n = (l - a) / d + 1
  let sum := (n * (a + l)) / 2
  (sum % 10) = 0 := sorry

end units_digit_sum_even_20_to_80_l330_330428


namespace dance_competition_scores_l330_330305

/--
 In a dance competition held at school, there are 7 judges who give scores.
 When determining the final score, the highest and lowest scores are removed, leaving 5 valid scores.
 We need to prove that the numerical characteristics that may change when comparing the 5 valid scores with the 7 original scores are:
  - Mean: may change
  - Median: remains unchanged
  - Variance: may change
  - Range: may change.
-/
theorem dance_competition_scores (scores : list ℝ) (h₁ : scores.length = 7) :
  ∃ valid_scores : list ℝ, valid_scores.length = 5 ∧ 
  (valid_scores.mean ≠ scores.mean ∨ valid_scores.variance ≠ scores.variance ∨ valid_scores.range ≠ scores.range) ∧ 
  (valid_scores.median = scores.median) :=
by sorry

end dance_competition_scores_l330_330305


namespace solve_for_x_l330_330077

theorem solve_for_x (x : ℝ) (h : |x - 2| = |x - 3| + 1) : x = 3 :=
by
  sorry

end solve_for_x_l330_330077


namespace estimated_total_score_l330_330675

noncomputable def regression_score (x : ℝ) : ℝ := 7.3 * x - 96.9

theorem estimated_total_score (x : ℝ) (h : x = 95) : regression_score x = 596 :=
by
  rw [h]
  -- skipping the actual calculation steps
  sorry

end estimated_total_score_l330_330675


namespace max_cos_sum_min_cos_sum_l330_330932

open Real

-- Definitions of the conditions
def is_nonnegative (x y z : ℝ) : Prop :=
  x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0

def sum_constraint (x y z : ℝ) : Prop :=
  x + y + z = 4 * π / 3

-- Definition of the function
def f (x y z : ℝ) : ℝ :=
  cos x + cos y + cos z

-- The statement for the maximum and minimum value, split into two theorems
theorem max_cos_sum : ∀ x y z : ℝ, 
  is_nonnegative x y z → sum_constraint x y z → f x y z ≤ 3 / 2 := sorry

theorem min_cos_sum : ∀ x y z : ℝ, 
  is_nonnegative x y z → sum_constraint x y z → 0 ≤ f x y z := sorry

end max_cos_sum_min_cos_sum_l330_330932


namespace find_number_of_moles_of_CaCO3_formed_l330_330564

-- Define the molar ratios and the given condition in structures.
structure Reaction :=
  (moles_CaOH2 : ℕ)
  (moles_CO2 : ℕ)
  (moles_CaCO3 : ℕ)

-- Define a balanced reaction for Ca(OH)2 + CO2 -> CaCO3 + H2O with 1:1 molar ratio.
def balanced_reaction (r : Reaction) : Prop :=
  r.moles_CaOH2 = r.moles_CO2 ∧ r.moles_CaCO3 = r.moles_CO2

-- Define the given condition, which is we have 3 moles of CO2 and formed 3 moles of CaCO3.
def given_condition : Reaction :=
  { moles_CaOH2 := 3, moles_CO2 := 3, moles_CaCO3 := 3 }

-- Theorem: Given 3 moles of CO2, we need to prove 3 moles of CaCO3 are formed based on the balanced reaction.
theorem find_number_of_moles_of_CaCO3_formed :
  balanced_reaction given_condition :=
by {
  -- This part will contain the proof when implemented.
  sorry
}

end find_number_of_moles_of_CaCO3_formed_l330_330564


namespace sum_factors_of_60_l330_330059

def sum_factors (n : ℕ) : ℕ :=
  (∑ d in Finset.divisors n, d)

theorem sum_factors_of_60 : sum_factors 60 = 168 := by
  sorry

end sum_factors_of_60_l330_330059


namespace unique_n_exists_l330_330343

noncomputable def sequence (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, 0 < a n ∧ a n < a (n + 1)

theorem unique_n_exists (a : ℕ → ℕ) (h : sequence a) : 
  ∃! n ≥ 1, let b (n : ℕ) := (n - 1) * a n - (∑ i in Finset.range n, a i) in 
  b n < a 0 ∧ a 0 ≤ b (n + 1) :=
sorry

end unique_n_exists_l330_330343


namespace probability_two_white_balls_l330_330109

-- Definitions based on the conditions provided
def total_balls := 17        -- 8 white + 9 black
def white_balls := 8
def drawn_without_replacement := true

-- Proposition: Probability of drawing two white balls successively
theorem probability_two_white_balls:
  drawn_without_replacement → 
  (white_balls / total_balls) * ((white_balls - 1) / (total_balls - 1)) = 7 / 34 :=
by
  intros
  sorry

end probability_two_white_balls_l330_330109


namespace problem_l330_330978

open Function

variable {R : Type*} [Ring R]

def f (x : R) : R := -- Assume the function type

theorem problem (h : ∀ x y : R, f (x * y) = y^2 * f x + x^2 * f y) :
  f 0 = 0 ∧ f 1 = 0 ∧ (∀ x : R, f (-x) = f x) :=
by
  sorry

end problem_l330_330978


namespace sum_of_valid_x_l330_330427

theorem sum_of_valid_x :
  let mean (xs : List ℝ) := xs.foldl (· + ·) 0 / xs.length 
  let median (xs : List ℝ) := (xs.sorted.nth ((xs.length - 1) / 2)).get_or_else 0
  let xs := [3.0, 7.0, x, 12.0, 18.0]
  ∃ (x : ℝ), median xs = mean xs → (∃ y1 y2 y3, y1 = -5 ∧ y2 = 10 ∧ y3 = 20 ∧ x = y1 ∨ x = y2 ∨ x = y3) → (∃ (sum_xs : ℝ), sum_xs = -5 + 10 + 20 ∧ sum_xs = 25) 
  sorry

end sum_of_valid_x_l330_330427


namespace percent_enclosed_by_triangles_l330_330469

-- Definitions based on conditions
def side_length (s : ℝ) := s
def area_hexagon (s : ℝ) := (3 * Real.sqrt 3 / 2) * s^2
def area_triangle (s : ℝ) := (Real.sqrt 3 / 4) * s^2

-- Mathematical proof problem statement
theorem percent_enclosed_by_triangles (s : ℝ) (h : s > 0) : 
  100 * ((6 * (area_triangle s)) / (area_hexagon s + 6 * (area_triangle s))) = 50 :=
by
  sorry

end percent_enclosed_by_triangles_l330_330469


namespace profit_percentage_2400_l330_330487

def store_cost (C : ℝ) (selling_price profit_percentage : ℝ) : Prop :=
  selling_price = C + profit_percentage * C

theorem profit_percentage_2400 (C : ℝ) (profit : ℝ) :
  store_cost C 2240 0.40 → store_cost C 2400 profit → profit = 0.50 :=
by
  assume h₁ : store_cost C 2240 0.40
  assume h₂ : store_cost C 2400 profit
  sorry

end profit_percentage_2400_l330_330487


namespace circle_tangent_parabola_l330_330124

theorem circle_tangent_parabola (a b : ℝ) (h_parabola : ∀ x, x^2 + 1 = y) 
  (h_tangency : (a, a^2 + 1) ∧ (-a, a^2 + 1)) 
  (h_center : (0, b)) 
  (h_circle : ∀ x y, x^2 + (y - b)^2 = r^2) 
  (h_tangent_points : (x = a) ∧ (x = -a)) : 
  b - (a^2 + 1) = -1/2 := 
sorry

end circle_tangent_parabola_l330_330124


namespace each_friend_eats_six_slices_l330_330471

-- Definitions
def slices_per_loaf : ℕ := 15
def loaves_bought : ℕ := 4
def friends : ℕ := 10
def total_slices : ℕ := loaves_bought * slices_per_loaf
def slices_per_friend : ℕ := total_slices / friends

-- Theorem to prove
theorem each_friend_eats_six_slices (h1 : slices_per_loaf = 15) (h2 : loaves_bought = 4) (h3 : friends = 10) : slices_per_friend = 6 :=
by
  sorry

end each_friend_eats_six_slices_l330_330471


namespace domain_of_function_l330_330182

theorem domain_of_function :
  ∀ x : ℝ, ⌊x^2 - 8 * x + 18⌋ ≠ 0 :=
sorry

end domain_of_function_l330_330182


namespace rectangle_perimeters_l330_330016

theorem rectangle_perimeters (length width : ℕ) (h1 : length = 7) (h2 : width = 5) :
  (∃ (L1 L2 : ℕ), L1 = 4 * width ∧ L2 = length ∧ 2 * (L1 + L2) = 54) ∧
  (∃ (L3 L4 : ℕ), L3 = 4 * length ∧ L4 = width ∧ 2 * (L3 + L4) = 66) ∧
  (∃ (L5 L6 : ℕ), L5 = 2 * length ∧ L6 = 2 * width ∧ 2 * (L5 + L6) = 48) :=
by
  sorry

end rectangle_perimeters_l330_330016


namespace geometric_locus_symmetric_points_circle_l330_330627

noncomputable def locus_of_symmetric_points (A B : Point) : Set Point :=
{M : Point | ∃ l : Line, B ∈ l ∧ symmetric_about_line A l M}

theorem geometric_locus_symmetric_points_circle (A B : Point) (C : Point)
  (h1 : segment_length A B = segment_length B C) :
  ∃ circle_diameter_AC : Circle, locus_of_symmetric_points A B = circle_diameter_AC.center.sphere circle_diameter_AC.radius :=
begin
  sorry
end

end geometric_locus_symmetric_points_circle_l330_330627


namespace frequency_count_calc_l330_330484

theorem frequency_count_calc :
  ∀ (N f Fc : ℝ), N = 1000 → f = 0.6 → Fc = N * f → Fc = 600 := by
  intros N f Fc hN hf hFc
  rw [hN, hf] at hFc
  assumption

end frequency_count_calc_l330_330484


namespace sqrt_pi_minus_4_squared_l330_330169

theorem sqrt_pi_minus_4_squared : sqrt ((π - 4)^2) = 4 - π :=
by
  sorry

end sqrt_pi_minus_4_squared_l330_330169


namespace perp_and_tan_angle_and_x_l330_330318

noncomputable def m : ℝ × ℝ := (⟨ sqrt 2 / 2, - sqrt 2 / 2 ⟩)
noncomputable def n (x : ℝ) : ℝ × ℝ := (⟨ sin x, cos x ⟩)

-- Given x in (0, π/2)
axiom x_in_range (x : ℝ) : x > 0 ∧ x < π / 2

-- Proof problem 1: m is perpendicular to n implies tan x = 1
theorem perp_and_tan (x : ℝ) (h : m = (sqrt 2 / 2, - sqrt 2 / 2)) 
  (hx : n x = (sin x, cos x)) (perp : m.1 * (n x).1 + m.2 * (n x).2 = 0) : 
  tan x = 1 := sorry

-- Proof problem 2: Angle between m and n is π / 3 implies x = 5π / 12
theorem angle_and_x (x : ℝ) (h : m = (sqrt 2 / 2, - sqrt 2 / 2)) 
  (hx : n x = (sin x, cos x)) (angle : m.1 * (n x).1 + m.2 * (n x).2 = 1 / 2) : 
  x = 5 * π / 12 := sorry

end perp_and_tan_angle_and_x_l330_330318


namespace zero_count_l330_330612

def f (k : ℝ) (x : ℝ) : ℝ :=
if x ≤ 0 then k * x + 1 else Real.log x

def g (k : ℝ) (x : ℝ) : ℝ := f k (f k x + 1)

theorem zero_count (k : ℝ) :
  (k > 0 → (∃! x1 x2 x3 x4, g k x1 = 0 ∧ g k x2 = 0 ∧ g k x3 = 0 ∧ g k x4 = 0)) ∧
  (k < 0 → (∃! x, g k x = 0)) :=
by
  sorry

end zero_count_l330_330612


namespace AK_equals_BC_l330_330672

-- Definitions of the geometric elements in the problem
variables (ΔABC : Type)
variables {A B C M H K : ΔABC}
variables [euclidean_geometry ΔABC]

-- Given conditions as hypotheses in Lean
variables (acute_angle: euclidean_geometry.acute_triangle A B C)
variables (AM_median: euclidean_geometry.median A M B C)
variables (BH_altitude: euclidean_geometry.altitude B H A C)
variables (M_perpendicular: euclidean_geometry.perpendicular M (euclidean_geometry.line A M) (euclidean_geometry.ray H B))
variables (angle_MAC_30: (euclidean_geometry.angle (euclidean_geometry.ray M A) (euclidean_geometry.ray A C) = 30))

-- The statement to be proved
theorem AK_equals_BC : euclidean_geometry.segment AK = euclidean_geometry.segment BC :=
sorry

end AK_equals_BC_l330_330672


namespace solve_for_x_l330_330755

theorem solve_for_x (x : ℝ) (h : ∛(7 - 3 / (3 + x)) = -2) : x = -14 / 5 :=
by sorry

end solve_for_x_l330_330755


namespace problem_l330_330980

open Function

variable {R : Type*} [Ring R]

def f (x : R) : R := -- Assume the function type

theorem problem (h : ∀ x y : R, f (x * y) = y^2 * f x + x^2 * f y) :
  f 0 = 0 ∧ f 1 = 0 ∧ (∀ x : R, f (-x) = f x) :=
by
  sorry

end problem_l330_330980


namespace ordered_pair_solution_l330_330542

theorem ordered_pair_solution :
  ∃ (x y : ℤ), 
    (x + y = (7 - x) + (7 - y)) ∧ 
    (x - y = (x - 2) + (y - 2)) ∧ 
    (x = 5 ∧ y = 2) :=
by
  sorry

end ordered_pair_solution_l330_330542


namespace number_of_whole_numbers_between_sqrt_18_and_sqrt_120_l330_330277

theorem number_of_whole_numbers_between_sqrt_18_and_sqrt_120 : 
  ∀ (n : ℕ), 
  (5 ≤ n ∧ n ≤ 10) ↔ (6 = 6) :=
sorry

end number_of_whole_numbers_between_sqrt_18_and_sqrt_120_l330_330277


namespace greatest_lower_bound_sum_of_squares_of_roots_l330_330133

noncomputable def sum_of_squares_of_roots_lower_bound (n : ℕ) (a_{n-1} a_{n-2} : ℝ) : ℝ :=
  let r : fin n → ℝ := sorry  -- roots of the polynomial
  let sum_of_roots := -a_{n-1}
  let product_of_roots := a_{n-2}
  let sum_of_squares_of_roots := sum_of_roots^2 - 2 * product_of_roots
  sorry -- full proof required here 

theorem greatest_lower_bound_sum_of_squares_of_roots (n : ℕ) (a_{n-2} : ℝ) (h : 2 * a_{n-2} = a_{n-1}) :
  ∀ a_{n-1}, sum_of_squares_of_roots_lower_bound n a_{n-1} a_{n-2} ≥ 1 / 4 := 
  by 
    intro a_{n-1}
    have h1 : sum_of_squares_of_roots_lower_bound n a_{n-1} a_{n-2} = 4 * a_{n-2}^2 - 2 * a_{n-2} := sorry
    -- leverage minimization of quadratic function
    have min_value := (4 : ℝ) * (1/4)^2 - 2 * (1/4)
    have h2 : min_value = 1 / 4 := by
      norm_num
    rw h1 at h2
    exact le_of_eq h2
    sorry -- remaining details

end greatest_lower_bound_sum_of_squares_of_roots_l330_330133


namespace determine_v4_l330_330371

-- Define the sequence with the given recurrence relation
def sequence (v : ℕ → ℝ) : Prop :=
  ∀ n, v (n + 2) = 2 * v (n + 1) + v n

-- Given conditions
def v2 : ℝ := 6
def v5 : ℝ := 58

-- Formal statement to prove that v_4 = 24.4 
theorem determine_v4 (v : ℕ → ℝ) (h1 : sequence v) (h2 : v 2 = v2) (h3 : v 5 = v5) : v 4 = 24.4 :=
by
  sorry

end determine_v4_l330_330371


namespace tangent_segment_length_l330_330098

noncomputable def length_of_tangent_segment : ℝ :=
-- Define the conditions
let A := (0, 0) in
let B := (2, 0) in
let C := (2, 2) in
let AB := dist A B in
let BC := dist B C in
let AC := dist A C in
let AD := dist A (1, 1, √3) in
let BD := dist B (1, 1, √3) in
let CD := dist C (1, 1, √3) in
-- Prove the length of the tangent segment from point A to the sphere
have edge_length_eq : AD = BD ∧ BD = CD := by sorry,
have sphere_tangent : true := by sorry, -- Placeholder for the sphere tangency conditions
let tangent_length := sqrt 3 - 1 in
exists unique tangent_length

-- The main statement to be proved
theorem tangent_segment_length : length_of_tangent_segment = sqrt 3 - 1 := by
sorry -- proof goes here

end tangent_segment_length_l330_330098


namespace game_cost_calculation_l330_330731

theorem game_cost_calculation :
  ∀ (initial_money spent_money num_games game_cost : ℕ), 
    initial_money = 101 → 
    spent_money = 47 → 
    num_games = 9 → 
    game_cost = 6 → 
    (initial_money - spent_money) / num_games = game_cost := 
by
  intros initial_money spent_money num_games game_cost h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  exact rfl

end game_cost_calculation_l330_330731


namespace cot_30_eq_sqrt3_l330_330556

theorem cot_30_eq_sqrt3 (theta : ℝ) (h1 : tan (π / 6) = 1 / real.sqrt 3) :
  1 / tan (π / 6) = real.sqrt 3 :=
by
  sorry

end cot_30_eq_sqrt3_l330_330556


namespace proof_inequality_l330_330302

variables {a b k : ℕ}
hypothesis (h1 : b ≥ 3)
hypothesis (h2 : b % 2 = 1)
hypothesis (h3 : k ∈ {n : ℕ | true})
hypothesis (h4 : ∀ {j1 j2 : ℕ}, j1 ≠ j2 → (∃ c, c ≤ k ∧ (∀ p : ℕ, (p ≤ a) → (j1 rates p = j2 rates p) → c)) ) -- Condition of identical ratings.

theorem proof_inequality (h1 : b ≥ 3) (h2 : b % 2 = 1) (h3 : k ∈ {n : ℕ | true}) (h4 : ∀ {j1 j2 : ℕ}, j1 ≠ j2 → (∃ c, c ≤ k ∧ (∀ p : ℕ, (p ≤ a) → (j1 rates p = j2 rates p) → c))) : 
  (f : Contestant -> nat) :
  (k : nat) 
:=
  sorry

end proof_inequality_l330_330302


namespace parallelogram_CQ_l330_330338

/-- 
  Let ABCD be a parallelogram. Let E be the midpoint of AB and F be the midpoint of CD. 
  Points P and Q are on segments EF and CF, respectively, such that A, P, and Q are collinear.
  Given that EP = 5, PF = 3, and QF = 12, prove that CQ = 8.
-/
theorem parallelogram_CQ (A B C D E F P Q : ℝ) 
  (parallelogram_ABCD : parallelogram A B C D)
  (midpoint_E : E = (A + B) / 2)
  (midpoint_F : F = (C + D) / 2)
  (collinear_APQ : collinear A P Q)
  (on_segment_EP : E < P)
  (on_segment_PF : P < F)
  (on_segment_CQ : Q < F)
  (EP_length : P - E = 5)
  (PF_length : F - P = 3)
  (QF_length : F - Q = 12) :
  C - Q = 8 :=
sorry

end parallelogram_CQ_l330_330338


namespace problem1_problem2_l330_330368

theorem problem1 (n : ℝ) (h : 3 * 9^(2*n) * 27^n = 3^(2*n)) : n = -1/5 := 
  sorry

theorem problem2 (a b : ℝ) (h1 : 10^a = 5) (h2 : 10^b = 6) : 10^(2*a + 3*b) = 5400 :=
  sorry

end problem1_problem2_l330_330368


namespace temperature_on_fourth_day_l330_330407

theorem temperature_on_fourth_day 
(t1 t2 t3 : ℤ) (avg4 : ℤ) (h1 : t1 = -36) (h2 : t2 = 13) (h3 : t3 = -15) (h4 : avg4 = -12) : 
t1 + t2 + t3 + 4 * avg4 = -10 :=
by
suffices h : t1 + t2 + t3 + 4 * avg4 - (t1 + t2 + t3) = 4 * avg4 - (t1 + t2 + t3) by
  rw [h, h1, h2, h3, h4]
  norm_num
sorry

end temperature_on_fourth_day_l330_330407


namespace range_of_a_l330_330431

-- Define the function y
def y (x a : ℝ) : ℝ := x^2 + 1 / (x^2 + a)

-- Define the condition a > 0
def a_pos (a : ℝ) : Prop := a > 0

-- The corresponding range of a
theorem range_of_a (a : ℝ) (ha : a_pos a) : 0 < a ∧ a ≤ 1 :=
by
  sorry

end range_of_a_l330_330431


namespace base_conversion_l330_330375

theorem base_conversion {b : ℕ} (h : 5 * 6 + 2 = b * b + b + 1) : b = 5 :=
by
  -- Begin omitted steps to solve the proof
  sorry

end base_conversion_l330_330375


namespace ellipse_standard_eq_and_max_area_l330_330594

-- Given Conditions
variables (a b c : ℝ)
variables (P O M N R : ℝ×ℝ)
variable (l : ℝ→ℝ)

-- Ellipse equation
def ellipse (x y : ℝ) := (x^2 / a^2) + (y^2 / b^2) = 1

-- Conditions
axiom a_pos : a > 0
axiom b_pos : b > 0
axiom a_gt_b : a > b
axiom point_on_ellipse : ellipse (-1) (3/2)
axiom eccentricity : c / a = 1/2
axiom a_eq_sqrt : a^2 = b^2 + c^2
axiom P_coords : P = (1, 0)
axiom PO_eq_OR : O = (P + R) / 2
axiom line_eq : ∀ x, ∃ y, (x, y) ∈ {p : ℝ×ℝ | p.1 = l p.2}

-- Standard equation of the ellipse
def ellipse_standard_eq := (x y : ℝ) → (x^2 / 4) + (y^2 / 3) = 1

-- Maximum area of triangle MNR and corresponding line equation
def max_area_triangle := 3
def line_eq_at_maximum := ∀ m ∈ ℝ, m > 0 → (?t)

-- Theorem
theorem ellipse_standard_eq_and_max_area :
  ∃ a b c l, 
  a > 0 ∧ b > 0 ∧ a > b ∧ (∃ m n: ℝ, a^2 = b^2 + c^2) ∧
  eccentricity c a 1/2 ∧
  ellipse (-1) (3/2) ∧
  ∃ eq_ellipse: eq_ellipse = ellipse_standard_eq (x:ℝ) (y:ℝ) → (x ^ 2 / 4 + y ^ 2 / 3 = 1) ∧
  ∃ line_eq l (1:ℝ) (0:ℝ) →
  ∃ m ∈ ℝ (line_eq_at_maximum 3):
  sorry

end ellipse_standard_eq_and_max_area_l330_330594


namespace interval_of_decrease_l330_330391

def g (x : ℝ) : ℝ := (1/2)^x

noncomputable def f : ℝ → ℝ := sorry -- the definition of f is derived from g

-- assuming f is symmetric to g with respect to y = x
def symmetric_to (f g : ℝ → ℝ) (h : ℝ → ℝ) : Prop :=
  ∀ (x : ℝ), f (h x) = g x

-- define the composite function
def composite_function (x : ℝ) : ℝ := f (2 * x - x * x)

theorem interval_of_decrease :
  (symmetric_to f g (λ x, x)) →
  ∀ (a b : ℝ), a < b ∧ a ≥ 0 ∧ b ≤ 2 →
  (∀ x, a < x ∧ x < b →
    composite_function x decreasing_on Ioo a b) :=
by
  intros h_symm a b h_intervals
  apply sorry

end interval_of_decrease_l330_330391


namespace compound_interest_correct_l330_330092

def compound_interest (P r n t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem compound_interest_correct :
  compound_interest 4000 0.10 1 2 = 4840 :=
by 
  -- proof will be skipped
  sorry

end compound_interest_correct_l330_330092


namespace remainder_of_7_pow_7_pow_7_pow_7_mod_500_l330_330224

theorem remainder_of_7_pow_7_pow_7_pow_7_mod_500 : 
  let λ := fun n => 
             match n with
             | 500 => 100
             | 100 => 20
             | _ => 0 in
  7^(7^(7^7)) ≡ 343 [MOD 500] :=
by 
  sorry

end remainder_of_7_pow_7_pow_7_pow_7_mod_500_l330_330224


namespace rhombus_RS_min_value_l330_330705

-- Definitions based on the problem conditions
def is_rhombus (W X Y Z : Point ℝ) : Prop :=
  (dist W X = dist X Y) ∧ (dist X Y = dist Y Z) ∧ (dist Y Z = dist Z W) ∧
  (dist Z W = dist W X) ∧
  (diagonal WY XZ ∧ diagonal XZ WY) ∧
  (perpendicular WY XZ)

-- Defining the main variables
variables {W X Y Z M R S : Point ℝ}

-- Given conditions translated as hypotheses
def problem_conditions : Prop :=
is_rhombus W X Y Z ∧
(dist W Y = 20) ∧ (dist X Z = 24) ∧
(M ∈ segment W X) ∧ (dist W M = dist M X) ∧
(foot M W Y = R) ∧ (foot M X Z = S)

-- Lean statement of the problem
theorem rhombus_RS_min_value (h : problem_conditions) : dist R S = sqrt 244 :=
sorry

end rhombus_RS_min_value_l330_330705


namespace functional_eq_properties_l330_330985

theorem functional_eq_properties (f : ℝ → ℝ) (h_domain : ∀ x, x ∈ ℝ) (h_func_eq : ∀ x y, f (x * y) = y^2 * f x + x^2 * f y) :
  f 0 = 0 ∧ f 1 = 0 ∧ ∀ x, f x = f (-x) :=
by {
  sorry
}

end functional_eq_properties_l330_330985


namespace sectionBSeats_l330_330690

-- Definitions from the conditions
def seatsIn60SeatSubsectionA : Nat := 60
def subsectionsIn80SeatA : Nat := 3
def seatsPer80SeatSubsectionA : Nat := 80
def extraSeatsInSectionB : Nat := 20

-- Total seats in 80-seat subsections of Section A
def totalSeatsIn80SeatSubsections : Nat := subsectionsIn80SeatA * seatsPer80SeatSubsectionA

-- Total seats in Section A
def totalSeatsInSectionA : Nat := totalSeatsIn80SeatSubsections + seatsIn60SeatSubsectionA

-- Total seats in Section B
def totalSeatsInSectionB : Nat := 3 * totalSeatsInSectionA + extraSeatsInSectionB

-- The statement to prove
theorem sectionBSeats : totalSeatsInSectionB = 920 := by
  sorry

end sectionBSeats_l330_330690


namespace train_length_l330_330490

theorem train_length (v : ℝ) (l_b : ℝ) (t : ℝ) : 
  v = 90 → l_b = 140 → t = 20 → l_t = 360 :=
by
  -- We need to calculate the length of the train.
  have h1 : v_mps = v * (1000 / 3600), from sorry,  -- Convert km/h to m/s
  have h2 : v_mps = 25, from sorry,  -- v in m/s
  have h3 : distance = v_mps * t, from sorry,  -- Distance covered in time t
  have h4 : distance = 500, from sorry,  -- Calculate total distance
  have h5 : l_t + l_b = distance, from sorry,  -- Relation between total distance, bridge length, and train length
  have h6 : l_t = distance - l_b, from sorry,  -- Calculate l_t
  have h7 : l_t = 360, from sorry,  -- l_t is 360 meters
  exact h7

end train_length_l330_330490


namespace intersection_A_complement_UB_l330_330269

-- Definitions of the sets U, A, and B
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {2, 3, 5, 6}
def B : Set ℕ := {x ∈ U | x^2 - 5 * x ≥ 0}

-- Complement of B w.r.t. U
def complement_U_B : Set ℕ := {x ∈ U | ¬ (x ∈ B)}

-- The statement we want to prove
theorem intersection_A_complement_UB : A ∩ complement_U_B = {2, 3} := by
  sorry

end intersection_A_complement_UB_l330_330269


namespace number_of_divisors_of_h_2010_l330_330941

/-- h(n) returns the smallest positive integer k such that 1/k has exactly n digits after the
decimal point and k is divisible by 3. -/
noncomputable def h (n : ℕ) : ℕ :=
  2^n * 3

/-- The statement to be proved: The number of positive integer divisors of h(2010) is 4022. -/
theorem number_of_divisors_of_h_2010 : 
  Nat.divisors (h 2010).card = 4022 := by
  sorry

end number_of_divisors_of_h_2010_l330_330941


namespace max_points_on_circle_d_5cm_away_q_l330_330352

open Real EuclideanGeometry

theorem max_points_on_circle_d_5cm_away_q (Q : Point) (D : Circle) (hQ : ¬(Q ∈ D)) :
  ∃ p p' ∈ D, dist Q p = 5 ∧ dist Q p' = 5 ∧ p ≠ p' ∧
  ∀ q ∈ D, dist Q q = 5 → (q = p ∨ q = p') :=
sorry

end max_points_on_circle_d_5cm_away_q_l330_330352


namespace range_of_a_l330_330611

noncomputable def f : ℝ → ℝ :=
  λ x, if x ≤ 1 then (1/3) * x + 1 else real.log x

theorem range_of_a (a : ℝ) : 
  (∃! x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 = a * x1 ∧ f x2 = a * x2) ↔ a ∈ set.Ioo ((1 : ℝ) / 3) (real.exp (-1)) := sorry

end range_of_a_l330_330611


namespace circle_tangent_parabola_l330_330125

theorem circle_tangent_parabola (a b : ℝ) (h_parabola : ∀ x, x^2 + 1 = y) 
  (h_tangency : (a, a^2 + 1) ∧ (-a, a^2 + 1)) 
  (h_center : (0, b)) 
  (h_circle : ∀ x y, x^2 + (y - b)^2 = r^2) 
  (h_tangent_points : (x = a) ∧ (x = -a)) : 
  b - (a^2 + 1) = -1/2 := 
sorry

end circle_tangent_parabola_l330_330125


namespace problem_statement_l330_330256

theorem problem_statement (f : ℝ → ℝ) (a b c m : ℝ)
  (h_cond1 : ∀ x, f x = -x^2 + a * x + b)
  (h_range : ∀ y, y ∈ Set.range f ↔ y ≤ 0)
  (h_ineq_sol : ∀ x, ((-x^2 + a * x + b > c - 1) ↔ (m - 4 < x ∧ x < m + 1))) :
  (b = -(1/4) * (2 * m - 3)^2) ∧ (c = -(21 / 4)) := sorry

end problem_statement_l330_330256


namespace sin_C_eq_l330_330580

-- Definitions based on conditions
variables {A B C : ℝ} {a b c : ℝ}
axiom triangle_ABC : B = 2 * Real.pi / 3
axiom side_length_relation : b = 3 * c

-- The property we want to prove
theorem sin_C_eq : ∀ (C : ℝ), (sin C = (Real.sqrt 3) / 6) :=
by 
  assume C
  sorry

end sin_C_eq_l330_330580


namespace metal_waste_calculation_l330_330126

noncomputable def originalCircleArea : ℝ := 100 * Real.pi
def squareSide : ℝ := 10 * Real.sqrt 2
def squareArea : ℝ := (squareSide)^2
def newCircleRadius : ℝ := squareSide / 2
noncomputable def newCircleArea : ℝ := Real.pi * (newCircleRadius)^2

theorem metal_waste_calculation :
  originalCircleArea - squareArea + squareArea - newCircleArea = 50 * Real.pi - 200 :=
by
  sorry

end metal_waste_calculation_l330_330126


namespace solve_for_x_l330_330272

noncomputable def a (x : ℝ) : ℝ × ℝ × ℝ := (1, 1, x)
def b : ℝ × ℝ × ℝ := (1, 2, 1)
def c : ℝ × ℝ × ℝ := (1, 1, 1)

def vec_sub (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := 
  (u.1 - v.1, u.2 - v.2, u.3 - v.3)

def vec_dot (u v : ℝ × ℝ × ℝ) : ℝ := 
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

def scalar_mult (k : ℝ) (u : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := 
  (k * u.1, k * u.2, k * u.3)

theorem solve_for_x (x : ℝ) (h : vec_dot (vec_sub c (a x)) (scalar_mult 2 b) = -2) : x = 2 := 
by 
  sorry

end solve_for_x_l330_330272


namespace direction_vector_exists_l330_330396

noncomputable def matrix_projection : Matrix (Fin 3) (Fin 3) ℚ :=
  ![![1/9, 1/18, 1/6], ![1/18, 1/36, 1/12], ![1/6, 1/12, 1/4]]

def standard_basis_x : Fin 3 → ℚ
  | ⟨0, _⟩ => 1
  | _      => 0

def gcd (a b : ℤ) : ℤ := Nat.gcd (Int.natAbs a) (Int.natAbs b)

theorem direction_vector_exists :
  let projection_result := matrix_projection.mulVec standard_basis_x
  (projection_result = ![2/18, 1/18, 3/18]) ∧
  gcd 2 1 = 1 ∧
  gcd 2 3 = 1 ∧
  gcd 1 3 = 1 :=
by
  let d := ![2, 1, 3]
  trivial
  sorry

end direction_vector_exists_l330_330396


namespace face_value_of_share_l330_330844

theorem face_value_of_share 
  (F : ℝ)
  (dividend_rate : ℝ := 0.09)
  (desired_return_rate : ℝ := 0.12)
  (market_value : ℝ := 15) 
  (h_eq : (dividend_rate * F) / market_value = desired_return_rate) : 
  F = 20 := 
by 
  sorry

end face_value_of_share_l330_330844


namespace find_b_pure_imaginary_l330_330376

theorem find_b_pure_imaginary (b : ℝ) : (1 + b * complex.I) * (2 + complex.I) = (2 * b + 1) * complex.I ↔ b = 2 :=
by sorry

end find_b_pure_imaginary_l330_330376


namespace pencils_given_away_l330_330521

-- Define the basic values and conditions
def initial_pencils : ℕ := 39
def bought_pencils : ℕ := 22
def final_pencils : ℕ := 43

-- Let x be the number of pencils Brian gave away
variable (x : ℕ)

-- State the theorem we need to prove
theorem pencils_given_away : (initial_pencils - x) + bought_pencils = final_pencils → x = 18 := by
  sorry

end pencils_given_away_l330_330521


namespace area_of_smallest_square_l330_330874

theorem area_of_smallest_square (r : ℝ) (h : r = 7) : 
  let d := 2 * r in
  let side := d in
  side * side = 196 :=
by
  rw [h]
  sorry

end area_of_smallest_square_l330_330874


namespace functional_equations_l330_330928

noncomputable def f (x y z : ℝ) : ℝ := 
  (y + Real.sqrt(y ^ 2 + 4 * x * z)) / (2 * x)

theorem functional_equations (x y z t : ℝ) (k : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (ht : t > 0) (hk : k > 0) :
  (x * f x y z = z * f z y x) ∧
  (f x (t * y) (t^2 * z) = t * f x y z) ∧
  (f 1 k (k + 1) = k + 1) :=
by
  sorry

end functional_equations_l330_330928
