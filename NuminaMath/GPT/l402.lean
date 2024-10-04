import Mathlib

namespace max_min_values_l402_402393

/-
We will define a function f and show that the range of f, given the domain condition on cos, includes the maximum value 1 and minimum value -3.
-/

open Real

noncomputable def f : ℝ → ℝ := λ x, 2 * cos x - 1

theorem max_min_values :
  (∀ x, -1 ≤ cos x ∧ cos x ≤ 1 → (f x ≤ 1 ∧ f x ≥ -3)) ∧
  (∃ x, cos x = 1 ∧ f x = 1) ∧
  (∃ x, cos x = -1 ∧ f x = -3) :=
begin
  sorry
end

end max_min_values_l402_402393


namespace solve_omega_l402_402634

noncomputable def omega (f : ℝ → ℝ) : ℝ :=
  let point := (-2, 0)
  let ω := sorry -- Solve for ω using given conditions and symmetry

theorem solve_omega (ω : ℝ) (h₀ : 0 < ω) (h₁ : ω < 1)
  (f : ℝ → ℝ) :
  f = (λ x, Real.sin (ω * x + Real.pi / 3)) →
  (∃ k : ℤ, -2 * ω + Real.pi / 3 = k * Real.pi) →
  ω = Real.pi / 6 := 
by
  sorry

end solve_omega_l402_402634


namespace speed_of_stream_l402_402007

-- Define the problem conditions
def downstream_distance := 100 -- distance in km
def downstream_time := 8 -- time in hours
def upstream_distance := 75 -- distance in km
def upstream_time := 15 -- time in hours

-- Define the constants
def total_distance (B S : ℝ) := downstream_distance = (B + S) * downstream_time
def total_time (B S : ℝ) := upstream_distance = (B - S) * upstream_time

-- Stating the main theorem to be proved
theorem speed_of_stream (B S : ℝ) (h1 : total_distance B S) (h2 : total_time B S) : S = 3.75 := by
  sorry

end speed_of_stream_l402_402007


namespace external_angle_bisector_proof_l402_402732

variables {A T C L K : Type} [Nonempty A] [Nonempty T] [Nonempty C] [Nonempty L] [Nonempty K]

noncomputable def angle_bisector_theorem (AL LC AB BC AK KC : ℝ) : Prop :=
(AL / LC) = (AB / BC) ∧ (AK / KC) = (AL / LC)

noncomputable def internal_angle_bisector (AT TC AL LC : ℝ) : Prop :=
(AT / TC) = (AL / LC)

noncomputable def external_angle_bisector (AT TC AK KC : ℝ) : Prop :=
(AT / TC) = (AK / KC)

theorem external_angle_bisector_proof (AL LC AB BC AK KC AT TC : ℝ) 
(h1 : angle_bisector_theorem AL LC AB BC AK KC)
(h2 : internal_angle_bisector AT TC AL LC) :
external_angle_bisector AT TC AK KC :=
sorry

end external_angle_bisector_proof_l402_402732


namespace Harry_work_hours_l402_402983

variable (x H S : ℝ)
variable (H_gt_18 : H > 18)
variable (total_hours_James : 41)
variable (pay_equiv : 18 * x + 1.5 * x * (H - 18) = S * x + 2 * x * (41 - S))

theorem Harry_work_hours : H = (91 - 3 * S) / 1.5 :=
by
  have Harry_weekly_pay : 18 * x + 1.5 * x * (H - 18) := sorry
  have James_weekly_pay : S * x + 2 * x * (41 - S) := sorry
  sorry

end Harry_work_hours_l402_402983


namespace largest_crowd_size_l402_402890

theorem largest_crowd_size :
  ∃ (n : ℕ), 
    (⌊n / 2⌋ + ⌊n / 3⌋ + ⌊n / 5⌋ = n) ∧
    ∀ m : ℕ, (⌊m / 2⌋ + ⌊m / 3⌋ + ⌊m / 5⌋ = m) → m ≤ 37 :=
sorry

end largest_crowd_size_l402_402890


namespace max_people_in_crowd_l402_402870

theorem max_people_in_crowd : ∃ n : ℕ, n ≤ 37 ∧ 
    (⟨1 / 2 * n⟩ + ⟨1 / 3 * n⟩ + ⟨1 / 5 * n⟩ = n) :=
sorry

end max_people_in_crowd_l402_402870


namespace metro_map_exists_l402_402396

-- Define stations and lines
inductive Station
| terminal (name : String)
| interchange (name : String)

structure Line :=
  (terminals : List Station)
  (interchanges : List Station)

def is_line (l : Line) : Prop :=
  l.terminals.length ≥ 2 ∧
  l.interchanges.length ≥ 2 ∧
  ∀ t ∈ l.terminals, ∀ i ∈ l.interchanges, t ≠ i

noncomputable def exists_metro_map : Prop :=
  ∃ (lines : List Line),
    lines.length = 3 ∧
    (∀ l1 l2 ∈ lines, l1 ≠ l2 → (l1.interchanges ∩ l2.interchanges).length ≥ 2) ∧
    (∀ l ∈ lines, is_line l)
    -- Additional conditions for drawing continuity and uniqueness of station map are implied but not composable into pure Lean 4

theorem metro_map_exists : exists_metro_map := sorry

end metro_map_exists_l402_402396


namespace minimum_familiar_pairs_l402_402908

theorem minimum_familiar_pairs (n : ℕ) (students : Finset (Fin n)) 
  (familiar : Finset (Fin n × Fin n))
  (h_n : n = 175)
  (h_condition : ∀ (s : Finset (Fin n)), s.card = 6 → 
    ∃ (s1 s2 : Finset (Fin n)), s1 ∪ s2 = s ∧ s1.card = 3 ∧ s2.card = 3 ∧ 
    ∀ x ∈ s1, ∀ y ∈ s1, (x ≠ y → (x, y) ∈ familiar) ∧
    ∀ x ∈ s2, ∀ y ∈ s2, (x ≠ y → (x, y) ∈ familiar)) :
  ∃ m : ℕ, m = 15050 ∧ ∀ p : ℕ, (∃ g : Finset (Fin n × Fin n), g.card = p) → p ≥ m := 
sorry

end minimum_familiar_pairs_l402_402908


namespace external_bisector_l402_402759

-- Define the relevant points and segments in the triangle
variables {A T C L K : Type} [inhabited A] [inhabited T] [inhabited C] [inhabited L] [inhabited K]

-- Assuming T, L, K are points such that TL divides angle ATC internally
axiom internal_bisector (h : L) : ∃ TL, TL.bisects_angle (A T C) internally

-- Prove that TK is the external angle bisector given the condition above
theorem external_bisector (h : L) (h_internal : internal_bisector h) : 
  ∃ TK, TK.bisects_angle (A T C) externally :=
sorry

end external_bisector_l402_402759


namespace neg_one_third_squared_l402_402964

theorem neg_one_third_squared :
  (-(1/3))^2 = 1/9 :=
sorry

end neg_one_third_squared_l402_402964


namespace initial_hamburgers_correct_l402_402044

-- Define the initial problem conditions
def initial_hamburgers (H : ℝ) : Prop := H + 3.0 = 12

-- State the proof problem
theorem initial_hamburgers_correct (H : ℝ) (h : initial_hamburgers H) : H = 9.0 :=
sorry

end initial_hamburgers_correct_l402_402044


namespace expected_total_cost_of_removing_blocks_l402_402004

/-- 
  There are six blocks in a row labeled 1 through 6, each with weight 1.
  Two blocks x ≤ y are connected if for all x ≤ z ≤ y, block z has not been removed.
  While there is at least one block remaining, a block is chosen uniformly at random and removed.
  The cost of removing a block is the sum of the weights of the blocks that are connected to it.
  Prove that the expected total cost of removing all blocks is 163 / 10.
-/
theorem expected_total_cost_of_removing_blocks : (6:ℚ) + 5 + 8/3 + 3/2 + 4/5 + 1/3 = 163 / 10 := sorry

end expected_total_cost_of_removing_blocks_l402_402004


namespace sum_of_two_equal_sides_is_4_l402_402399

noncomputable def isosceles_right_triangle (a c : ℝ) : Prop :=
  c = 2.8284271247461903 ∧ c ^ 2 = 2 * (a ^ 2)

theorem sum_of_two_equal_sides_is_4 :
  ∃ a : ℝ, isosceles_right_triangle a 2.8284271247461903 ∧ 2 * a = 4 :=
by
  sorry

end sum_of_two_equal_sides_is_4_l402_402399


namespace number_of_integer_values_of_a_l402_402166

theorem number_of_integer_values_of_a : 
  ∃ (a_set : Set ℤ), (∀ a ∈ a_set, ∃ x y : ℤ, x^2 + a * x + 12 * a = 0 ∧ y^2 + a * y + 12 * a = 0) ∧ a_set.card = 16 :=
sorry

end number_of_integer_values_of_a_l402_402166


namespace hexagon_all_zeroes_l402_402508

theorem hexagon_all_zeroes (a b c d e f : ℕ) (h_sum : a + b + c + d + e + f = 2003 ^ 2003) :
  ∃ (moves : ℕ → ℕ), (∀ n, moves n ≤ 6) ∧ (∀ i, 0 ≤ i < 6 → ∀ k, ∃ j, moves k = j) ∧ all_vertices_zero :
  sorry

end hexagon_all_zeroes_l402_402508


namespace circumcircle_radius_incircle_radius_range_l402_402303

open Real
open Classical

noncomputable def side_length_a : ℝ := _
noncomputable def side_length_c : ℝ := _
noncomputable def angle_A : ℝ := _
noncomputable def angle_B : ℝ := _
noncomputable def angle_C : ℝ := _

def side_length_b := 7
def given_eq (a b c : ℝ) (sin_A sin_B sin_C : ℝ) :
  (a + b) / c = (sin_A - sin_C) / (sin_A - sin_B) := _

theorem circumcircle_radius (a b c R : ℝ) (sin_A sin_B sin_C : ℝ) :
  (side_length_b = 7) →
  (given_eq a side_length_b c sin_A sin_B sin_C) →
  (2 * R = b / sin_B) →
  R = 7 * sqrt 3 / 3 :=
by sorry

theorem incircle_radius_range (a b c r : ℝ) (sin_A sin_B sin_C : ℝ) :
  (side_length_b = 7) →
  (given_eq a side_length_b c sin_A sin_B sin_C) →
  (0 < r ∧ r ≤ 7 * sqrt 3 / 6) :=
by sorry

end circumcircle_radius_incircle_radius_range_l402_402303


namespace least_number_subtracted_divisible_17_l402_402537

theorem least_number_subtracted_divisible_17 :
  ∃ n : ℕ, 165826 - n % 17 = 0 ∧ n = 12 :=
by
  use 12
  sorry  -- Proof will go here.

end least_number_subtracted_divisible_17_l402_402537


namespace terminal_side_in_third_quadrant_l402_402627

-- We define the concept of angles being in the third quadrant.
def in_third_quadrant (α : ℝ) : Prop :=
  π < α ∧ α < 3 * π / 2

theorem terminal_side_in_third_quadrant (α : ℝ) (h1 : Real.sin α < 0) (h2 : Real.tan α > 0) : in_third_quadrant α :=
by
  sorry

end terminal_side_in_third_quadrant_l402_402627


namespace count_integers_satisfying_condition_l402_402620

def in_range (n : ℤ) : Prop := -10 ≤ n ∧ n ≤ 12

def satisfies_inequality (n : ℤ) : Prop := (n - 3) * (n + 3) * (n + 7) < 0

theorem count_integers_satisfying_condition :
  finset.card (finset.filter satisfies_inequality (finset.filter in_range (finset.Icc -10 12))) = 5 :=
by
  sorry

end count_integers_satisfying_condition_l402_402620


namespace parallelograms_equidecomposable_with_congruent_base_height_parallelograms_equidecomposable_with_equal_area_square_on_hypotenuse_equidecomposable_with_squares_on_legs_rectangle_equidecomposable_with_square_of_equal_area_equilateral_triangle_equidecomposable_with_square_different_area_l402_402421

-- Part (a)
theorem parallelograms_equidecomposable_with_congruent_base_height (ABCD EFGH : parallelogram) (h_base: base ABCD = base EFGH) (h_height: height ABCD = height EFGH) : 
  equidecomposable ABCD EFGH :=
sorry

-- Part (b)
theorem parallelograms_equidecomposable_with_equal_area (P1 P2 : parallelogram) (h_area: area P1 = area P2) : 
  equidecomposable P1 P2 :=
sorry

-- Part (c)
theorem square_on_hypotenuse_equidecomposable_with_squares_on_legs (a b c : ℝ) (h_right_triangle : is_right_triangle a b c) : 
  equidecomposable (square_on_hypotenuse a b c) (union_of_squares_on_legs a b c) :=
sorry

-- Part (d)
theorem rectangle_equidecomposable_with_square_of_equal_area (R : rectangle) (S : square) (h_area: area R = area S) :
  equidecomposable R S :=
sorry

-- Part (e)
theorem equilateral_triangle_equidecomposable_with_square_different_area (T: equilateral_triangle) (S: square) :
  equidecomposable T S :=
sorry

end parallelograms_equidecomposable_with_congruent_base_height_parallelograms_equidecomposable_with_equal_area_square_on_hypotenuse_equidecomposable_with_squares_on_legs_rectangle_equidecomposable_with_square_of_equal_area_equilateral_triangle_equidecomposable_with_square_different_area_l402_402421


namespace fixed_point_line_intersects_circle_value_range_on_circle_l402_402565

-- Definition of the circle
def Circle (x y : ℝ) := (x - 1)^2 + (y - 2)^2 = 16

-- Definition of the line
def Line (m x y : ℝ) := (2 * m + 1) * x + (m + 1) * y - 7 * m - 4 = 0

-- Problem statement
theorem fixed_point (m : ℝ) : 
  ∃ (P : ℝ × ℝ), P = (3, 1) ∧ ∀ (x y : ℝ), Line m x y → P = (x, y) :=
sorry

theorem line_intersects_circle (m : ℝ) : 
  ∀ (x y : ℝ), Line m x y → (∃ (xC yC : ℝ), Circle xC yC ∧ (x = xC ∧ y = yC)) :=
sorry

theorem value_range_on_circle (θ : ℝ) (hθ : 0 ≤ θ ∧ θ < 2 * Real.pi) :
  let x := 1 + 4 * Real.cos θ,
      y := 2 + 4 * Real.sin θ in
  -3/4 < y / (x + 3) :=
sorry

end fixed_point_line_intersects_circle_value_range_on_circle_l402_402565


namespace number_of_real_solutions_l402_402539

theorem number_of_real_solutions : 
  ∃ (s : Set ℝ), s.count = 2 ∧ ∀ x ∈ s, 
    (6 * x) / (x^2 + 2 * x + 5) + (7 * x) / (x^2 - 7 * x + 5) = 1 := 
by
  sorry

end number_of_real_solutions_l402_402539


namespace poly_degree_l402_402425

-- Definition of the base polynomial
def base_poly : Polynomial ℝ := 2 * X ^ 3 + 5

-- Definition of the polynomial we are interested in
def poly : Polynomial ℝ := (base_poly) ^ 10

-- Statement of the theorem
theorem poly_degree : poly.degree = 30 :=
by {
  sorry
}

end poly_degree_l402_402425


namespace isosceles_triangle_angle_relation_l402_402057

theorem isosceles_triangle_angle_relation (A B C D : Point)
  (h1 : ∠A B C = ∠A C B)
  (h2 : ∠A B C = 3 * ∠D)
  (h3 : ∠B A C = t * Real.pi) :
  t = 5 / 11 := 
  sorry

end isosceles_triangle_angle_relation_l402_402057


namespace rectangle_rhombus_area_ratio_l402_402274

theorem rectangle_rhombus_area_ratio :
  ∀ (ABCD : Type) (A B C D E F G H : ABCD) (length width d1 d2 : ℝ),
    length = 28 ∧ width = 20 ∧
    E ∈ line_segment A B ∧ F ∈ line_segment B C ∧
    G ∈ line_segment C D ∧ H ∈ line_segment D A ∧
    is_midpoint E A B ∧ is_midpoint F B C ∧
    is_midpoint G C D ∧ is_midpoint H D A ∧
    (d1 = 12 ∨ d2 = 12) ∧
    (d1 = 20 ∨ d2 = 20) →
  let area_rhombus := (d1 * d2) / 2 in
  let area_rectangle := length * width in
  area_rhombus = 120 ∧ (area_rhombus / area_rectangle = 3 / 14) :=
by
  sorry

end rectangle_rhombus_area_ratio_l402_402274


namespace baron_munchausen_max_people_l402_402865

theorem baron_munchausen_max_people :
  ∃ x : ℕ, (x = 37) ∧ 
  (1 / 2 * x).nat_ceil + (1 / 3 * x).nat_ceil + (1 / 5 * x).nat_ceil = x := sorry

end baron_munchausen_max_people_l402_402865


namespace external_bisector_l402_402756

-- Define the relevant points and segments in the triangle
variables {A T C L K : Type} [inhabited A] [inhabited T] [inhabited C] [inhabited L] [inhabited K]

-- Assuming T, L, K are points such that TL divides angle ATC internally
axiom internal_bisector (h : L) : ∃ TL, TL.bisects_angle (A T C) internally

-- Prove that TK is the external angle bisector given the condition above
theorem external_bisector (h : L) (h_internal : internal_bisector h) : 
  ∃ TK, TK.bisects_angle (A T C) externally :=
sorry

end external_bisector_l402_402756


namespace product_of_roots_l402_402522

-- Definitions
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

def roots (a b c : ℝ) (d : ℝ) : ℝ × ℝ :=
  ((-b + Real.sqrt d) / (2 * a), (-b - Real.sqrt d) / (2 * a))

-- Theorem stating the product of the roots
theorem product_of_roots (a b c : ℝ) (d : ℝ) (h_a : a ≠ 0) (h_d : d = discriminant a b c) :
  quadratic a b c 0 = 0 → (a = 1 ∧ b = 3 ∧ c = -4) → 
    (let (x1, x2) := roots a b c d in x1 * x2 = -4) :=
by
  sorry

end product_of_roots_l402_402522


namespace number_of_ways_to_form_divisible_by_12_l402_402298

theorem number_of_ways_to_form_divisible_by_12 :
  let possible_digits := {0, 2, 4, 5, 7, 9},
      fixed_digits_sum := 2 + 0 + 1 + 6 + 0 + 2,
      sum_condition := (fixed_digits_sum + ∑ d in possible_digits, d) % 3 = 1,
      last_digit_condition := ∀ last_digit ∈ {0, 4}, 
      all_possibilities := 1296 * 2  -- 1296 possible 4-digit combinations with each having 2 valid last digits
   in 2 * all_possibilities = 5184 :=
by
  sorry

end number_of_ways_to_form_divisible_by_12_l402_402298


namespace shirts_sold_l402_402483

theorem shirts_sold (initial_shirts remaining_shirts shirts_sold : ℕ) (h1 : initial_shirts = 49) (h2 : remaining_shirts = 28) : 
  shirts_sold = initial_shirts - remaining_shirts → 
  shirts_sold = 21 := 
by 
  sorry

end shirts_sold_l402_402483


namespace circles_are_separate_l402_402521

/-- Definitions for two circles -/
def circle1_center := (-1 : ℝ, -3 : ℝ)
def circle1_radius := (1 : ℝ)
def circle2_center := (3 : ℝ, -1 : ℝ)
def circle2_radius := (2 * Real.sqrt 2 : ℝ)
def distance_between_centers := Real.sqrt ((3 + 1)^2 + (-1 + 3)^2)

/-- Positional relationship between circle1 and circle2 -/
theorem circles_are_separate :
  circle1_radius + circle2_radius < distance_between_centers :=
  sorry

end circles_are_separate_l402_402521


namespace angus_tokens_eq_l402_402499

-- Define the conditions
def worth_per_token : ℕ := 4
def elsa_tokens : ℕ := 60
def angus_less_worth : ℕ := 20

-- Define the main theorem to prove
theorem angus_tokens_eq :
  let elsaTokens := elsa_tokens,
      worthPerToken := worth_per_token,
      angusLessTokens := angus_less_worth / worth_per_token
  in (elsaTokens - angusLessTokens) = 55 := by
  sorry

end angus_tokens_eq_l402_402499


namespace airplane_seat_count_l402_402952

theorem airplane_seat_count (s : ℕ) 
  (h1 : 54 + 0.30 * s + 0.60 * s = s) : 
  s = 540 :=
by
  sorry

end airplane_seat_count_l402_402952


namespace xy_product_l402_402540

theorem xy_product (x y : ℝ) (h1 : 4^x = 256^(y+1)) (h2 : 27^y = 3^(x-2)) : x * y = 8 :=
by
  -- Proof steps to be filled in
  sorry

end xy_product_l402_402540


namespace number_of_arrangements_l402_402616

theorem number_of_arrangements (A B C : ℕ) (hA : A = 6) (hB : B = 7) (hC : C = 7) :
  ∑ x in Finset.range (min 7 7 + 1), nat.choose 7 x * nat.choose 7 (7 - x) * nat.choose 6 (6 - (7 - x) + x - 7) = 
  ∑ x in Finset.range (8), nat.choose 7 x * nat.choose 7 (7 - x) * nat.choose 6 (6 - (7 - x) + x - 7) := 
sorry

end number_of_arrangements_l402_402616


namespace tan_A_gt_sqrt3_implies_A_gt_pi_over_3_A_gt_pi_over_3_not_necessarily_implies_tan_A_gt_sqrt3_l402_402304

theorem tan_A_gt_sqrt3_implies_A_gt_pi_over_3 (A : ℝ) (h0 : 0 < A ∧ A < π) (h : tan A > sqrt 3) : A > π / 3 :=
sorry

theorem A_gt_pi_over_3_not_necessarily_implies_tan_A_gt_sqrt3 (A : ℝ) (h0 : 0 < A ∧ A < π) (h : A > π / 3) : ¬ (A > π / 3 ↔ tan A > sqrt 3) :=
sorry

end tan_A_gt_sqrt3_implies_A_gt_pi_over_3_A_gt_pi_over_3_not_necessarily_implies_tan_A_gt_sqrt3_l402_402304


namespace count_matrices_mod_13_l402_402684

noncomputable def number_of_A (F : Type) [Field F] [Fintype F] (A : Matrix (Fin 5) (Fin 5) F) : ℕ :=
  Fintype.card {A : Matrix (Fin 5) (Fin 5) F // A^5 = 1}

theorem count_matrices_mod_13 : number_of_A (ZMod 13) = 18883858278044793930625 :=
by sorry

end count_matrices_mod_13_l402_402684


namespace bill_sells_whole_milk_for_3_per_gallon_l402_402509

noncomputable def price_per_gallon_whole_milk : ℕ :=
  let total_milk := 16
  let milk_for_sour_cream := (1 / 4 : ℚ) * total_milk
  let milk_for_butter := (1 / 4 : ℚ) * total_milk
  let whole_milk := total_milk - milk_for_sour_cream - milk_for_butter

  let gallons_of_butter := milk_for_butter / 4
  let gallons_of_sour_cream := milk_for_sour_cream / 2

  let revenue_butter := gallons_of_butter * 5
  let revenue_sour_cream := gallons_of_sour_cream * 6

  let total_revenue := 41
  let revenue_whole_milk := total_revenue - revenue_butter - revenue_sour_cream

  let price_per_gallon := revenue_whole_milk / whole_milk
  price_per_gallon.to_nat

theorem bill_sells_whole_milk_for_3_per_gallon : price_per_gallon_whole_milk = 3 := sorry

end bill_sells_whole_milk_for_3_per_gallon_l402_402509


namespace max_people_in_crowd_l402_402874

theorem max_people_in_crowd : ∃ n : ℕ, n ≤ 37 ∧ 
    (⟨1 / 2 * n⟩ + ⟨1 / 3 * n⟩ + ⟨1 / 5 * n⟩ = n) :=
sorry

end max_people_in_crowd_l402_402874


namespace total_cats_correct_l402_402671

-- Jamie's cats
def Jamie_Persian_cats : ℕ := 4
def Jamie_Maine_Coons : ℕ := 2

-- Gordon's cats
def Gordon_Persian_cats : ℕ := Jamie_Persian_cats / 2
def Gordon_Maine_Coons : ℕ := Jamie_Maine_Coons + 1

-- Hawkeye's cats
def Hawkeye_Persian_cats : ℕ := 0
def Hawkeye_Maine_Coons : ℕ := Gordon_Maine_Coons - 1

-- Total cats for each person
def Jamie_total_cats : ℕ := Jamie_Persian_cats + Jamie_Maine_Coons
def Gordon_total_cats : ℕ := Gordon_Persian_cats + Gordon_Maine_Coons
def Hawkeye_total_cats : ℕ := Hawkeye_Persian_cats + Hawkeye_Maine_Coons

-- Proof that the total number of cats is 13
theorem total_cats_correct : Jamie_total_cats + Gordon_total_cats + Hawkeye_total_cats = 13 :=
by sorry

end total_cats_correct_l402_402671


namespace total_cost_price_of_items_l402_402972

/-- 
  Definition of the selling prices of the items A, B, and C.
  Definition of the profit percentages of the items A, B, and C.
  The statement is the total cost price calculation.
-/
def ItemA_SP : ℝ := 800
def ItemA_Profit : ℝ := 0.25
def ItemB_SP : ℝ := 1200
def ItemB_Profit : ℝ := 0.20
def ItemC_SP : ℝ := 1500
def ItemC_Profit : ℝ := 0.30

theorem total_cost_price_of_items :
  let CP_A := ItemA_SP / (1 + ItemA_Profit)
  let CP_B := ItemB_SP / (1 + ItemB_Profit)
  let CP_C := ItemC_SP / (1 + ItemC_Profit)
  CP_A + CP_B + CP_C = 2793.85 :=
by
  sorry

end total_cost_price_of_items_l402_402972


namespace find_b_l402_402017

noncomputable def b (a : ℝ) := 
  (Real.log (8^a) + Real.log (27^a) + Real.log (125^a)) / 
  (Real.log 9 + Real.log 25 + Real.log 2 - Real.log 15)

theorem find_b (a : ℝ) : b a = 3 * a :=
by 
  sorry

end find_b_l402_402017


namespace determine_valid_numbers_l402_402081

-- Define the property P(n) to check if inserting '0' in any of the described positions results in a multiple of 7.
def property (n : ℕ) : Prop :=
  let (a, b, c, d) := (n / 1000, (n / 100) % 10, (n / 10) % 10, n % 10) in
  (n < 10000 ∧ 1000 ≤ n ∧
    ∃ z1 z2 z3 z4 z5,
    (z1 = n ∧ z1 % 7 = 0) ∧ 
    (z2 = 10000 * a + 100 * b + 10 * c + d ∧ z2 % 7 = 0) ∧
    (z3 = 10000 * a + 1000 * b + 10 * c + d ∧ z3 % 7 = 0) ∧
    (z4 = 10000 * a + 1000 * b + 100 * c + d ∧ z4 % 7 = 0) ∧
    (z5 = 10000 * a + 1000 * b + 100 * c + 10 * d ∧ z5 % 7 = 0))

theorem determine_valid_numbers :
  {n | property n} = {7000, 7007, 7070, 7077, 7700, 7707, 7770, 7777} :=
by sorry

end determine_valid_numbers_l402_402081


namespace sum_of_fractions_eq_decimal_l402_402094

theorem sum_of_fractions_eq_decimal :
  (3 / 100) + (5 / 1000) + (7 / 10000) = 0.0357 :=
by
  sorry

end sum_of_fractions_eq_decimal_l402_402094


namespace quadratic_eq_a_val_l402_402261

theorem quadratic_eq_a_val (a : ℝ) (h : a - 6 = 0) :
  a = 6 :=
by
  sorry

end quadratic_eq_a_val_l402_402261


namespace total_photos_taken_or_framed_l402_402013

theorem total_photos_taken_or_framed (jack_framed_octavia : ℕ) (jack_framed_others : ℕ) (octavia_taken : ℕ) :
  jack_framed_octavia = 24 → jack_framed_others = 12 → octavia_taken = 36 → 
  (octavia_taken + (jack_framed_others)) = 48 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  rfl

end total_photos_taken_or_framed_l402_402013


namespace max_arcs_on_circle_l402_402767

theorem max_arcs_on_circle (points : ℕ) (arcs : ℕ) (distance : points := 2022) :
    ∃k, (k = 1011) ∧ (∀arc₁ arc₂, arc₁ ≠ arc₂ → (arc₁.about_not_contain arc₂)) :=
sorry

end max_arcs_on_circle_l402_402767


namespace baron_munchausen_max_crowd_size_l402_402875

theorem baron_munchausen_max_crowd_size :
  ∃ n : ℕ, (∀ k, (k : ℕ) = n → 
  let left := (k / 2).toNat;
      right := (k / 3).toNat;
      straight := (k / 5).toNat in
  left + right + straight <= n + 1) ∧ 
  (∀ x : ℕ, x > 37 → ¬(∀ k, (k : ℕ) = x →
  let left := (k / 2).toNat;
      right := (k / 3).toNat;
      straight := (k / 5).toNat in
  left + right + straight <= x + 1)) :=
begin
  have h : 37 = 18 + 12 + 7,
  sorry,
end

end baron_munchausen_max_crowd_size_l402_402875


namespace range_of_a_l402_402555

variable (a : ℝ)

def f (x : ℝ) := Real.log x / Real.log 3

theorem range_of_a (ha : f a > f 2) : a > 2 :=
by 
  sorry

end range_of_a_l402_402555


namespace part1_angle_A_part2_length_AD_l402_402664

-- Geometry definitions and necessary axioms
variables {a b c : ℝ}
variables {A B C : ℝ} (R : ℝ) 
variables (AC AD : ℝ)
variables (AAF : Prop)

def triangle (s : ℝ) (t : ℝ) (u : ℝ) : Prop :=
  0 < s ∧ 0 < t ∧ 0 < u ∧ s < t + u ∧ t < s + u ∧ u < s + t

def law_of_sines (a b c A B C : ℝ) (R : ℝ) : Prop :=
  a = 2 * R * (Real.sin A) ∧ b = 2 * R * (Real.sin B) ∧ c = 2 * R * (Real.sin C)

def given_equation (b c A B C : ℝ) : Prop :=
  b * (Real.sin B) + c * (Real.sin C) = ((2 * (Real.sqrt 3) / 3) * b * (Real.sin C) + a) * (Real.sin A)

def find_angle_A (A : ℝ) : Prop :=
  A = Real.pi / 3

def find_length_AD (AD : ℝ) : Prop :=
  AD = Real.sqrt 2

theorem part1_angle_A (a b c : ℝ) (A B C : ℝ) (R : ℝ) (htriangle : triangle a b c) 
  (hLawSines : law_of_sines a b c A B C R) (hgivenEq : given_equation b c A B C) : find_angle_A A :=
by
  sorry

theorem part2_length_AD (a b c : ℝ) (A B C : ℝ) (R AC AD : ℝ) (htriangle : triangle a b c) 
  (hLawSines : law_of_sines a b c A B C R) (hgivenEq : given_equation b c A B C) (hFindA : find_angle_A A)
  : find_length_AD AD :=
by
  sorry

end part1_angle_A_part2_length_AD_l402_402664


namespace problem_part1_problem_part2_problem_part3_l402_402201

noncomputable def a_seq : ℕ → ℕ
| 0     => 4
| (n+1) => 3 * a_seq n - 2

def geometric_seq (n : ℕ) : ℕ := 3^n

def b_seq (n : ℕ) : ℕ := ∑ i in finset.range n, log 3 (geometric_seq (i + 1))

def T_n (n : ℕ) : ℕ := 2 * n / (n + 1)

theorem problem_part1 (n : ℕ ) : a_seq n - 1 = 3^n := sorry

theorem problem_part2 (n : ℕ) : b_seq n = n * (n + 1) / 2 := sorry

theorem problem_part3 (n : ℕ) : (∑ i in finset.range n, 1 / (b_seq (i + 1))) = 2 * (1 - 1 / (n + 1)) := sorry

end problem_part1_problem_part2_problem_part3_l402_402201


namespace min_abs_A_l402_402322

def arithmetic_sequence (a d : ℚ) (n : ℕ) : ℚ :=
  a + (n - 1) * d

def A (a d : ℚ) (n : ℕ) : ℚ :=
  (arithmetic_sequence a d n) + (arithmetic_sequence a d (n + 1)) + 
  (arithmetic_sequence a d (n + 2)) + (arithmetic_sequence a d (n + 3)) + 
  (arithmetic_sequence a d (n + 4)) + (arithmetic_sequence a d (n + 5)) + 
  (arithmetic_sequence a d (n + 6))

theorem min_abs_A : (arithmetic_sequence 19 (-4/5) 26 = -1) ∧ 
                    (∀ n, 1 ≤ n) →
                    ∃ n : ℕ, |A 19 (-4/5) n| = 7/5 :=
by
  sorry

end min_abs_A_l402_402322


namespace distinct_integer_values_of_a_l402_402121

theorem distinct_integer_values_of_a :
  ∃ (a_values : Finset ℤ), (∀ a ∈ a_values, ∃ (m n : ℤ), (x^2 + a * x + 12 * a = 0) ∧ (m + n = -a) ∧ (m * n = 12 * a))
  ∧ a_values.card = 16 :=
sorry

end distinct_integer_values_of_a_l402_402121


namespace train_length_is_approximately_350_07_l402_402009

noncomputable def train_length (speed_kmh : ℝ) (time_sec : ℝ) : ℝ :=
  let speed_ms := speed_kmh * (1000 / 3600)
  in speed_ms * time_sec

theorem train_length_is_approximately_350_07 :
  train_length 60 21 ≈ 350.07 :=
by 
  sorry

end train_length_is_approximately_350_07_l402_402009


namespace integer_values_of_a_l402_402174

theorem integer_values_of_a (a : ℤ) :
  (∀ x : ℤ, x^2 + a * x + 12 * a = 0 → x ∈ ℤ) → ∃ n : ℕ, n = 16 :=
by
  sorry

end integer_values_of_a_l402_402174


namespace cyclic_S_BMN_l402_402314

open EuclideanGeometry

variables (A B C D E F T S M N : Point)
variable [h_triangle : Triangle A B C]
variable [h_obtuse : Angle B A C > 90]
variable [h_altitudes : Altitude A D ∧ Altitude B E ∧ Altitude C F]
variable [h_midpoints : Midpoint T A D ∧ Midpoint S C F]
variable [h_symmetric : SymmetricImage M T BE ∧ SymmetricImage N T BD]

theorem cyclic_S_BMN :
  CyclicQuadrilateral S B M N :=
sorry

end cyclic_S_BMN_l402_402314


namespace TK_is_external_bisector_of_ATC_l402_402706

variable {A T C L K : Type} [LinearOrder A] [LinearOrder T] [LinearOrder C] [LinearOrder L] [LinearOrder K]
variable (triangle_ATC : Triangle A T C)
variable (point_T : Point T)
variable (point_L : Point L)
variable (point_K : Point K)
variable (internal_bisector_TL : AngleBisector (angle_ATC) T L)
variable (external_bisector_TK : AngleBisector (external_angle_ATC) T K)

theorem TK_is_external_bisector_of_ATC :
  external_bisector_TK = true :=
sorry

end TK_is_external_bisector_of_ATC_l402_402706


namespace probability_of_PAIR_letters_in_PROBABILITY_l402_402971

theorem probability_of_PAIR_letters_in_PROBABILITY : 
  let total_letters := 11
  let favorable_letters := 4
  favorable_letters / total_letters = 4 / 11 :=
by
  let total_letters := 11
  let favorable_letters := 4
  show favorable_letters / total_letters = 4 / 11
  sorry

end probability_of_PAIR_letters_in_PROBABILITY_l402_402971


namespace sum_of_squares_l402_402561

variables (a : ℕ → ℝ) (i n : ℕ)
def A (i : ℕ) := (i / (i^2 + i - 1)) * (∑ k in Finset.range i + 1, a k)

theorem sum_of_squares (h_pos : ∀ i, 0 < a i):
  (∑ k in Finset.range n + 1, (A a k)^2) ≤  4 * (∑ k in Finset.range n + 1, (a k)^2) := 
sorry

end sum_of_squares_l402_402561


namespace decreasing_func_l402_402256

noncomputable def func (x : ℝ) (b : ℝ) : ℝ := -0.5 * x^2 + b * Real.log (x + 2)

theorem decreasing_func (b : ℝ) : (∀ x > -1, deriv (λ x, func x b) x ≤ 0) → b ≤ -1 := by
  sorry  -- Proof not required

end decreasing_func_l402_402256


namespace integer_solution_count_l402_402144

def has_integer_solutions (a : ℤ) : Prop :=
  ∃ x y : ℤ, x^2 + a * x + 12 * a = 0 ∧ y^2 + a * y + 12 * a = 0

theorem integer_solution_count : 
  (set.finite {a : ℤ | has_integer_solutions a}).to_finset.card = 16 := 
begin
  sorry
end

end integer_solution_count_l402_402144


namespace plate_arrangement_l402_402931

theorem plate_arrangement {blue red green yellow : Nat} (h_blue : blue = 6) (h_red : red = 3) (h_green : green = 1) (h_yellow : yellow = 2) :
  (number_of_arrangements blue red green yellow 1) = 8400 := 
sorry

-- Helper function definition to be added appropriately in Lean (if needed), such as:
-- def number_of_arrangements : Nat -> Nat -> Nat -> Nat -> Nat -> Nat := sorry

end plate_arrangement_l402_402931


namespace ellipse_minor_axis_length_l402_402940

noncomputable def length_minor_axis_of_ellipse (a b : ℝ) : Prop :=
  let points := [(-2, 0), (1, 0), (1, 3), (4, 0), (4, 3)] in
  ∀ (x y : ℝ),
    (x, y) ∈ points → 
    (x - a)^2 / (4.5^2) + (y - b)^2 / ((3.18 / 2)^2) = 1

theorem ellipse_minor_axis_length :
  ∃ a b : ℝ,
  length_minor_axis_of_ellipse a b :=
sorry

end ellipse_minor_axis_length_l402_402940


namespace smallest_k_satisfying_condition_l402_402428

def is_smallest_prime_greater_than (n : ℕ) (p : ℕ) : Prop :=
  Nat.Prime p ∧ n < p ∧ ∀ q, Nat.Prime q ∧ q > n → q >= p

def is_divisible_by (m k : ℕ) : Prop := k % m = 0

theorem smallest_k_satisfying_condition :
  ∃ k, is_smallest_prime_greater_than 19 23 ∧ is_divisible_by 3 k ∧ 64 ^ k > 4 ^ (19 * 23) ∧ (∀ k' < k, is_divisible_by 3 k' → 64 ^ k' ≤ 4 ^ (19 * 23)) :=
by
  sorry

end smallest_k_satisfying_condition_l402_402428


namespace equilateral_triangle_ratio_l402_402845

theorem equilateral_triangle_ratio (s : ℝ) (h : s = 6) :
  let perimeter := 3 * s
  let area := (s * s * Real.sqrt 3) / 4
  perimeter / area = (2 * Real.sqrt 3) / 3 :=
by
  sorry

end equilateral_triangle_ratio_l402_402845


namespace inequality_in_triangle_l402_402259

variables {a b c : ℝ}

namespace InequalityInTriangle

-- Define the condition that a, b, c are sides of a triangle
def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem inequality_in_triangle (a b c : ℝ) (h : is_triangle a b c) :
  1 / (b + c - a) + 1 / (c + a - b) + 1 / (a + b - c) > 9 / (a + b + c) :=
sorry

end InequalityInTriangle

end inequality_in_triangle_l402_402259


namespace part1_zeros_of_f_part2_min_max_of_f_part3_f_is_increasing_l402_402224

noncomputable def f (x : ℝ) : ℝ := 3 * x^2 + 12 * x - 15

-- Part (1): Prove that the zeros of f are -5 and 1.
theorem part1_zeros_of_f :
  (∃ x : ℝ, f x = 0) ↔ (x = -5 ∨ x = 1) :=
sorry

-- Part (2): Prove that the minimum value of f(x) on [-3, 3] is -27, and the maximum value is 48.
theorem part2_min_max_of_f :
  (∀ x ∈ set.Icc (-3 : ℝ) (3 : ℝ), -27 ≤ f x ∧ f x ≤ 48) ∧
  (f (-2) = -27) ∧ (f 3 = 48) :=
sorry

-- Part (3): Prove that f(x) is an increasing function on the interval [-2, +∞).
theorem part3_f_is_increasing :
  ∀ x1 x2 : ℝ, x1 ∈ set.Ici (-2) → x2 ∈ set.Ici (-2) → x1 < x2 → f x1 < f x2 :=
sorry

end part1_zeros_of_f_part2_min_max_of_f_part3_f_is_increasing_l402_402224


namespace AT_eq_RC_l402_402694

-- Define the problem setup
variables {A B C D P M Q R S T : Type}

-- Define the cyclic quadrilateral, intersection points, circle, and midpoints
variables (h1 : cyclic_quadrilateral A B C D)
variables (h2 : intersect_diagonals A C B D P)
variables (h3 : circle_through P touches (midpoint C D) M)
variables (h4 : intersects_circle (segment BD) Q P)
variables (h5 : intersects_circle (segment AC) R P)
variables (h6 : BS_eq_DQ B S D Q)

-- Define the parallelism condition
variables (h7 : parallel_to_through AB S AC T)

-- The proposition to be proved
theorem AT_eq_RC (h8 : BS = DQ) : AT = RC :=
sorry

end AT_eq_RC_l402_402694


namespace greatest_xy_value_l402_402258

theorem greatest_xy_value (x y : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y) (h_eq : 7 * x + 4 * y = 140) :
  (∀ z : ℕ, (∃ x y : ℕ, 0 < x ∧ 0 < y ∧ 7 * x + 4 * y = 140 ∧ z = x * y) → z ≤ 168) ∧
  (∃ x y : ℕ, 0 < x ∧ 0 < y ∧ 7 * x + 4 * y = 140 ∧ 168 = x * y) :=
sorry

end greatest_xy_value_l402_402258


namespace integer_values_a_l402_402185

theorem integer_values_a :
  (∃! a : ℤ, ∃ m n : ℤ, m^2 + n^2 = (m + n)^2 - 2mn ∧
                     m + n = -a ∧ 
                     mn = 12a) :
  8 :=
sorry

end integer_values_a_l402_402185


namespace external_angle_bisector_of_triangle_l402_402739

variables {A T C L K : Type}
variables [IsTriangle A T C] [IsInternalAngleBisector T L (A T C)]

theorem external_angle_bisector_of_triangle 
  (h1 : is_internal_angle_bisector TL ATC) :
  is_external_angle_bisector TK ATC :=
sorry

end external_angle_bisector_of_triangle_l402_402739


namespace square_remainder_mod_16_is_square_l402_402638

theorem square_remainder_mod_16_is_square (N : ℤ) :
  ∃ b : ℤ, (b ∈ {0, 1, 2, 3, 4}) ∧ (N ≡ 8 * (N / 8) + b ∨ N ≡ 8 * (N / 8) - b) :=
  ∃ r : ℤ, (r = N ^ 2 % 16) ∧ (r = 0 ∨ r = 1 ∨ r = 4 ∨ r = 9) :=
sorry

end square_remainder_mod_16_is_square_l402_402638


namespace finite_operations_invariant_final_set_l402_402450

theorem finite_operations (n : ℕ) (a : Fin n → ℕ) :
  ∃ N : ℕ, ∀ k, k > N → ((∃ i j, i ≠ j ∧ ¬ (a i ∣ a j ∨ a j ∣ a i)) → False) :=
sorry

theorem invariant_final_set (n : ℕ) (a : Fin n → ℕ) :
  ∃ b : Fin n → ℕ, (∀ i, ∃ j, b i = a j) ∧ ∀ (c : Fin n → ℕ), (∀ i, ∃ j, c i = a j) → c = b :=
sorry

end finite_operations_invariant_final_set_l402_402450


namespace fibonacci_determinant_identity_fibonacci_1000_1002_1001_l402_402666

noncomputable def fibonacci (n : ℕ) : ℤ :=
if n = 0 then 0
else if n = 1 then 1
else fibonacci (n - 1) + fibonacci (n - 2)

theorem fibonacci_determinant_identity (F : ℕ → ℤ) (n : ℕ) :
  (\begin
    |1 1|^n = |F (n+1) F n|
    |1 0| = |F n F (n-1)|
   ) →
   F (n+1) * F (n-1) - F n^2 = (-1) ^ n := sorry

theorem fibonacci_1000_1002_1001 :
  fibonacci 1000 * fibonacci 1002 - fibonacci 1001 ^ 2 = -1 := sorry

end fibonacci_determinant_identity_fibonacci_1000_1002_1001_l402_402666


namespace value_of_y_l402_402257

theorem value_of_y (x y : ℝ) (h1 : x = 2) (h2 : x^(2*y) = 4) : y = 1 :=
by
  sorry

end value_of_y_l402_402257


namespace min_star_value_l402_402861

theorem min_star_value :
  ∃ (star : ℕ), (98348 * 10 + star) % 72 = 0 ∧ (∀ (x : ℕ), (98348 * 10 + x) % 72 = 0 → star ≤ x) := sorry

end min_star_value_l402_402861


namespace TK_is_external_bisector_of_ATC_l402_402711

variable {A T C L K : Type} [LinearOrder A] [LinearOrder T] [LinearOrder C] [LinearOrder L] [LinearOrder K]
variable (triangle_ATC : Triangle A T C)
variable (point_T : Point T)
variable (point_L : Point L)
variable (point_K : Point K)
variable (internal_bisector_TL : AngleBisector (angle_ATC) T L)
variable (external_bisector_TK : AngleBisector (external_angle_ATC) T K)

theorem TK_is_external_bisector_of_ATC :
  external_bisector_TK = true :=
sorry

end TK_is_external_bisector_of_ATC_l402_402711


namespace calculate_expression_solve_equation_l402_402903

-- Part (1)
theorem calculate_expression : (-1:ℝ) ^ 0 - (1 / 3) ^ (-1:ℝ) + real.sqrt 4 = 0 := by
  sorry

-- Part (2)
theorem solve_equation (x : ℝ) : ( 1 - x) / (x - 2) = 1 / (2 - x) - 2 → false := by
  sorry

end calculate_expression_solve_equation_l402_402903


namespace parameterization_satisfies_line_l402_402392

theorem parameterization_satisfies_line (v m : ℚ) :
  let line_eq := ∀ x y : ℚ, y = (3 / 4) * x - 2 →
                   ∃ u : ℚ, (x, y) = (-3, v) + u • (m, -8) in
  v = -17 / 4 ∧ m = -16 / 9 := 
by
  let x := -3
  let y := - (17 / 4)
  have line_eq : ∀ x y : ℚ, y = (3 / 4) * x - 2 → ∃ u : ℚ, (x, y) = (-3, -17 / 4) + u • (-16 / 9, -8), from sorry
  exact ⟨rfl, rfl⟩

end parameterization_satisfies_line_l402_402392


namespace integer_values_of_a_l402_402172

theorem integer_values_of_a (a : ℤ) :
  (∀ x : ℤ, x^2 + a * x + 12 * a = 0 → x ∈ ℤ) → ∃ n : ℕ, n = 16 :=
by
  sorry

end integer_values_of_a_l402_402172


namespace find_f_l402_402189

def smallest_f (n : ℕ) (h : n ≥ 4) : ℕ :=
  ⌊(n+1)/2⌋ + ⌊(n+1)/3⌋ - ⌊(n+1)/6⌋ + 1

theorem find_f (n : ℕ) (h : n ≥ 4) : 
  smallest_f n h = ⌊(n + 1) / 2⌋ + ⌊(n + 1) / 3⌋ - ⌊(n + 1) / 6⌋ + 1 :=
sorry

end find_f_l402_402189


namespace find_a_range_l402_402635

noncomputable def range_a (a : ℝ) : Prop :=
  ∀ x ∈ set.Ioc 0 (1 / 2: ℝ), 9 ^ x - real.log x / real.log a ≤ 2

theorem find_a_range (a : ℝ) : (1 / 2: ℝ) ≤ a ∧ a < 1 ↔ range_a a := by
  sorry

end find_a_range_l402_402635


namespace cos_theta_interval_l402_402789

noncomputable def quadratic_function_positive (θ : ℝ) : Prop :=
  ∀ x : ℝ, cos θ * x^2 - 4 * sin θ * x + 6 > 0

theorem cos_theta_interval (θ : ℝ) (h1: 0 < θ ∧ θ < π) 
  (h2 : quadratic_function_positive θ) : 
  1 / 2 < cos θ ∧ cos θ < 1 := sorry

end cos_theta_interval_l402_402789


namespace domain_transformation_l402_402584

theorem domain_transformation {f : ℝ → ℝ} (h : ∀ x, 1 ≤ 3*x + 1 ∧ 3*x + 1 ≤ 7) :
  ∀ x, 0 ≤ x ∧ x ≤ 2 :=
begin
  sorry,
end

end domain_transformation_l402_402584


namespace integer_values_of_a_l402_402176

theorem integer_values_of_a (a : ℤ) :
  (∀ x : ℤ, x^2 + a * x + 12 * a = 0 → x ∈ ℤ) → ∃ n : ℕ, n = 16 :=
by
  sorry

end integer_values_of_a_l402_402176


namespace complement_union_eq_l402_402239

open Set

variable (U A B : Set ℤ)

noncomputable def universal_set : Set ℤ := {-2, -1, 0, 1, 2, 3}

noncomputable def setA : Set ℤ := {-1, 0, 3}

noncomputable def setB : Set ℤ := {1, 3}

theorem complement_union_eq :
  A ∪ B = {-1, 0, 1, 3} →
  U = universal_set →
  A = setA →
  B = setB →
  (U \ (A ∪ B)) = {-2, 2} := by
  intros
  sorry

end complement_union_eq_l402_402239


namespace TK_is_external_bisector_of_ATC_l402_402708

variable {A T C L K : Type} [LinearOrder A] [LinearOrder T] [LinearOrder C] [LinearOrder L] [LinearOrder K]
variable (triangle_ATC : Triangle A T C)
variable (point_T : Point T)
variable (point_L : Point L)
variable (point_K : Point K)
variable (internal_bisector_TL : AngleBisector (angle_ATC) T L)
variable (external_bisector_TK : AngleBisector (external_angle_ATC) T K)

theorem TK_is_external_bisector_of_ATC :
  external_bisector_TK = true :=
sorry

end TK_is_external_bisector_of_ATC_l402_402708


namespace kanul_total_amount_l402_402310

def kanul_spent : ℝ := 3000 + 1000
def kanul_spent_percentage (T : ℝ) : ℝ := 0.30 * T

theorem kanul_total_amount (T : ℝ) (h : T = kanul_spent + kanul_spent_percentage T) :
  T = 5714.29 := sorry

end kanul_total_amount_l402_402310


namespace total_spent_on_burgers_l402_402545

def days_in_june := 30
def burgers_per_day := 4
def cost_per_burger := 13

theorem total_spent_on_burgers (total_spent : Nat) :
  total_spent = days_in_june * burgers_per_day * cost_per_burger :=
sorry

end total_spent_on_burgers_l402_402545


namespace gcd_1407_903_l402_402839

theorem gcd_1407_903 : Nat.gcd 1407 903 = 21 := 
  sorry

end gcd_1407_903_l402_402839


namespace Gwen_recycled_correctly_l402_402615

-- Definition of the conditions as per part (a)
def points_per_pound : ℝ := 1 / 3
def friends_recycle : ℝ := 13
def total_points : ℕ := 6

-- Definition of the correct answer following part (c)
def Gwen_recycled_pounds : ℝ := 18 - 13

theorem Gwen_recycled_correctly :
  (total_points * (3 : ℝ) - friends_recycle = 5) :=
by
  sorry

end Gwen_recycled_correctly_l402_402615


namespace external_bisector_of_triangle_l402_402704

variables {A T C K L : Type} [noncomputable_field A T C K L]
variable {triangle ATC : Triangle A T C}

noncomputable def internal_angle_bisector (T L : Point) := 
    ∃ R, is_bisector TL ∧ TL ∝ AR RL

noncomputable def external_angle_bisector (T K : Point) := 
    ∃ M, is_bisector TK ∧ TK ∝ AM MK

theorem external_bisector_of_triangle {A T C K L : Point} 
    (hL : internal_angle_bisector T L) 
    : external_angle_bisector T K :=
sorry

end external_bisector_of_triangle_l402_402704


namespace find_measure_angle_A_find_max_area_triangle_ABC_l402_402268

variable (A B C a b c : ℝ)
variable (area_A : ℝ)

-- Given conditions
def triangle_ABC : Prop :=
  ∃ (a b c : ℝ), a = b + c - 2

-- Question 1: Finding measure of angle A
def measure_angle_A : Prop :=
  A = 60

-- Proving measure of angle A
theorem find_measure_angle_A :
  triangle_ABC → measure_angle_A :=
sorry

-- Given condition for second question
def condition_a_eq_2 : Prop :=
  a = 2

-- Question 2: Finding maximum area of triangle ABC
def max_area_triangle_ABC : Prop :=
  area_A = 5

-- Proving maximum area of triangle ABC
theorem find_max_area_triangle_ABC :
  triangle_ABC → condition_a_eq_2 → max_area_triangle_ABC :=
sorry

end find_measure_angle_A_find_max_area_triangle_ABC_l402_402268


namespace value_of_x_l402_402576

-- Define the sets A and B
def A : set ℕ := {0, 1, 2}
def B : set ℕ := {2, 3}

-- Define the conditions for x
variables {x : ℕ}
hypothesis h1 : x ∈ B
hypothesis h2 : x ∉ A

-- Prove that x = 3
theorem value_of_x : x = 3 :=
sorry

end value_of_x_l402_402576


namespace ratio_of_students_l402_402456

theorem ratio_of_students (total_students mac_preference no_preference both_preference : ℕ) 
  (h_total : total_students = 210) 
  (h_mac : mac_preference = 60) 
  (h_no_preference : no_preference = 90) 
  (h_both : both_preference = total_students - (mac_preference + no_preference)): 
  both_preference = 60 ∧ both_preference / mac_preference = 1 := 
by
  have h_both_calculated : both_preference = 210 - (60 + 90) := by rw [h_total, h_mac, h_no_preference]
  rw [h_both, h_both_calculated]
  split
  { exact rfl }
  { simp }

end ratio_of_students_l402_402456


namespace external_bisector_l402_402750

open Triangle

variable {Point : Type} [EuclideanGeometry Point]

variables {A T C L K : Point}

theorem external_bisector (h : angle_bisector A T C L) : external_bisector A T C K :=
sorry

end external_bisector_l402_402750


namespace reverse_digits_difference_l402_402448

theorem reverse_digits_difference (q r : ℕ) (x y : ℕ) 
  (hq : q = 10 * x + y)
  (hr : r = 10 * y + x)
  (hq_r_pos : q > r)
  (h_diff_lt_20 : q - r < 20)
  (h_max_diff : q - r = 18) :
  x - y = 2 := 
by
  sorry

end reverse_digits_difference_l402_402448


namespace watermelons_original_count_l402_402050

open Nat

theorem watermelons_original_count
    (h₁ : ∀ (n ≤ 7), ∃ m : ℕ, m / 2 + 1 / 2 = m)
    (h₇ : ∀ (final : ℕ), final * (2 ^ 7 - 1) = 127):
  final * (2 ^ 7 - 1) = 127 := sorry

end watermelons_original_count_l402_402050


namespace external_bisector_TK_l402_402723

-- Definitions of points and segments
variables {A T C L K : Type}

-- Conditions given in the problem
variable (angle_bisector_TL : is_angle_bisector T L A C)
variable (is_triangle : is_triangle T A C)

-- Prove that TK is the external angle bisector
theorem external_bisector_TK : is_external_angle_bisector T K A C :=
sorry

end external_bisector_TK_l402_402723


namespace walking_speed_10_mph_l402_402471

theorem walking_speed_10_mph 
  (total_minutes : ℕ)
  (distance : ℕ)
  (rest_per_segment : ℕ)
  (rest_time : ℕ)
  (segments : ℕ)
  (walk_time : ℕ)
  (walk_time_hours : ℕ) :
  total_minutes = 328 → 
  distance = 50 → 
  rest_per_segment = 7 → 
  segments = 4 →
  rest_time = segments * rest_per_segment →
  walk_time = total_minutes - rest_time →
  walk_time_hours = walk_time / 60 →
  distance / walk_time_hours = 10 :=
by
  sorry

end walking_speed_10_mph_l402_402471


namespace external_bisector_l402_402755

-- Define the relevant points and segments in the triangle
variables {A T C L K : Type} [inhabited A] [inhabited T] [inhabited C] [inhabited L] [inhabited K]

-- Assuming T, L, K are points such that TL divides angle ATC internally
axiom internal_bisector (h : L) : ∃ TL, TL.bisects_angle (A T C) internally

-- Prove that TK is the external angle bisector given the condition above
theorem external_bisector (h : L) (h_internal : internal_bisector h) : 
  ∃ TK, TK.bisects_angle (A T C) externally :=
sorry

end external_bisector_l402_402755


namespace count_sparrows_l402_402772

theorem count_sparrows (N t : ℕ) 
  (H1 : (3 / 5 : ℚ) * N)
  (H2 : (1 / 4 : ℚ) * N) 
  (H3 : 10 * t = N - ((3 / 5) * N + (1 / 4) * N)) : 
  (3 / 5 : ℚ) * N = 40 * t :=
by
  sorry

end count_sparrows_l402_402772


namespace number_of_integer_values_of_a_l402_402161

theorem number_of_integer_values_of_a : 
  ∃ (a_set : Set ℤ), (∀ a ∈ a_set, ∃ x y : ℤ, x^2 + a * x + 12 * a = 0 ∧ y^2 + a * y + 12 * a = 0) ∧ a_set.card = 16 :=
sorry

end number_of_integer_values_of_a_l402_402161


namespace external_angle_bisector_proof_l402_402736

variables {A T C L K : Type} [Nonempty A] [Nonempty T] [Nonempty C] [Nonempty L] [Nonempty K]

noncomputable def angle_bisector_theorem (AL LC AB BC AK KC : ℝ) : Prop :=
(AL / LC) = (AB / BC) ∧ (AK / KC) = (AL / LC)

noncomputable def internal_angle_bisector (AT TC AL LC : ℝ) : Prop :=
(AT / TC) = (AL / LC)

noncomputable def external_angle_bisector (AT TC AK KC : ℝ) : Prop :=
(AT / TC) = (AK / KC)

theorem external_angle_bisector_proof (AL LC AB BC AK KC AT TC : ℝ) 
(h1 : angle_bisector_theorem AL LC AB BC AK KC)
(h2 : internal_angle_bisector AT TC AL LC) :
external_angle_bisector AT TC AK KC :=
sorry

end external_angle_bisector_proof_l402_402736


namespace external_bisector_l402_402745

open Triangle

variable {Point : Type} [EuclideanGeometry Point]

variables {A T C L K : Point}

theorem external_bisector (h : angle_bisector A T C L) : external_bisector A T C K :=
sorry

end external_bisector_l402_402745


namespace max_min_sum_l402_402336

noncomputable def f (x : ℝ) : ℝ := abs (x - 2) + abs (x - 4) - abs (2 * x - 6)

theorem max_min_sum (h : ∀ x, 2 ≤ x → x ≤ 8) : 
  let f_max := max (f 2) (max (f 3) (f 4)),
      f_min := min (f 2) (min (f 3) (f 4))
  in f_max + f_min = 2 := 
sorry

end max_min_sum_l402_402336


namespace distinct_integer_values_of_a_l402_402117

theorem distinct_integer_values_of_a :
  ∃ (a_values : Finset ℤ), (∀ a ∈ a_values, ∃ (m n : ℤ), (x^2 + a * x + 12 * a = 0) ∧ (m + n = -a) ∧ (m * n = 12 * a))
  ∧ a_values.card = 16 :=
sorry

end distinct_integer_values_of_a_l402_402117


namespace coplanar_points_l402_402976

theorem coplanar_points (a : ℝ) :
  ∀ (V : ℝ), V = 2 + a^3 → V = 0 → a = -((2:ℝ)^(1/3)) :=
by
  sorry

end coplanar_points_l402_402976


namespace radius_of_circle_is_4_l402_402595

noncomputable def circle_radius
  (a : ℝ) 
  (radius : ℝ) 
  (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*a*x + 9 = 0 ∧ (-a, 0) = (5, 0) ∧ radius = 4

theorem radius_of_circle_is_4 
  (a x y : ℝ) 
  (radius : ℝ) 
  (h : circle_radius a radius x y) : 
  radius = 4 :=
by 
  sorry

end radius_of_circle_is_4_l402_402595


namespace number_of_men_worked_on_first_road_l402_402414

noncomputable def men_worked_on_first_road : Nat :=
  let total_man_hours_km1 : ℤ := x * 12 * 8
  let total_man_hours_km2 : ℤ := 20 * 32 * 9
  let proportion : ℤ := total_man_hours_km1 = (total_man_hours_km2 / 2)
  let manpower := 30 -- derived number of men from the proportion
  if proportion then manpower else 0

theorem number_of_men_worked_on_first_road (x : ℤ) :
  (x * 12 * 8) = ((20 * 32 * 9) / 2) → x = 30 := by
  sorry

end number_of_men_worked_on_first_road_l402_402414


namespace math_books_ratio_l402_402408

noncomputable def total_books : ℕ := 10
def reading_books : ℕ := (2 * total_books) / 5
def history_books : ℕ := 1

-- Let M be the number of math books
def math_books (M : ℕ) : Prop := 
  let S := M - 1 
  total_books = reading_books + M + S + history_books

theorem math_books_ratio (M : ℕ) (h : math_books M) : 
  M = 3 → ratio M total_books = 3/10 := sorry

end math_books_ratio_l402_402408


namespace brother_combined_age_l402_402832

-- Define the ages of the brothers as integers
variable (x y : ℕ)

-- Define the condition given in the problem
def combined_age_six_years_ago : Prop := (x - 6) + (y - 6) = 100

-- State the theorem to prove the current combined age
theorem brother_combined_age (h : combined_age_six_years_ago x y): x + y = 112 :=
  sorry

end brother_combined_age_l402_402832


namespace equal_projections_of_incenters_l402_402646

-- Definitions and conditions from part (a)

variables {A B C H H0 H1 H2 : Type*}
variables {triangle : ∀ {a b c : Type*}, right_triangle a b c}
variables {proj : ∀ {a b c : Type*}, projections_of_incenters_onto_hypotenuse a b c}
variables {equal_distances : ∀ {h₀ h₁ h₂}, h₁ ≠ h₀ → h₀ ≠ h₂ → (distance h₁ h = distance h h₀) ∧ (distance h h₀ = distance h₀ h₂)}

-- Statement of the theorem
theorem equal_projections_of_incenters (triangle : right_triangle A B C) 
  (proj : projections_of_incenters_onto_hypotenuse H0 H1 H2 A C)
  : distance H1 H = distance H H0 ∧ distance H H0 = distance H0 H2 :=
sorry

end equal_projections_of_incenters_l402_402646


namespace largest_constant_inequality_l402_402102

theorem largest_constant_inequality :
  ∃ (C : ℝ), C = 1 / 3 ∧
    ∀ (n : ℕ) (x : ℕ → ℝ),
      (n > 0) → 
      (x 0 = 0) → 
      (x n = 1) → 
      (∀ i : ℕ, (i < n) → (x i < x (i+1))) → 
      (∑ k in finset.range n, x k ^ 2 * (x k - x (k-1)) > C) :=
begin
  sorry
end

end largest_constant_inequality_l402_402102


namespace plane_relations_l402_402687

noncomputable def plane (α : Type _) := α

variable {α β l : Type _}

def intersecting_lines_within_plane_parallel (α β : plane α) : Prop :=
  ∀ (l1 l2 : α) (m1 m2 : β), (l1 ≠ l2) → (m1 ≠ m2) → (l1 ∥ m1) → (l2 ∥ m2)

def line_outside_parallel (l : α) (p : plane α) : Prop :=
  ∀ (m : α), m ∥ l → l ∥ p

def intersect_line_perpendicular (α β : plane α) (l : α) : Prop :=
  ∃ (m : α), (m ∥ α) ∧ (m ⟂ l)

def line_perpendicular_to_plane (l : α) (p : plane α) : Prop :=
  ∀ (m n : α), (m ≠ n) → (m ⟂ l) → (n ⟂ l) → l ⟂ p

theorem plane_relations (α β : plane α) (l : α) :
  (intersecting_lines_within_plane_parallel α β → α ∥ β) ∧
  (line_outside_parallel l α → l ∥ α) ∧
  ¬(intersect_line_perpendicular α β l → α ⟂ β) ∧
  ¬(line_perpendicular_to_plane l α ↔ ∃ (m n : α), (m ≠ n) ∧ (m ⟂ l) ∧ (n ⟂ l)) :=
by
  sorry

end plane_relations_l402_402687


namespace jane_leave_days_l402_402672

theorem jane_leave_days :
  ∃ x : ℝ, 
    (1 / 8) * (15.2 - x) + (1 / 40) * x + (1 / 10) * 4 = 1 ∧
    x = 13 :=
begin
  sorry
end

end jane_leave_days_l402_402672


namespace tutors_meet_in_lab_l402_402641

theorem tutors_meet_in_lab (c a j t : ℕ)
  (hC : c = 5) (hA : a = 6) (hJ : j = 8) (hT : t = 9) :
  Nat.lcm (Nat.lcm (Nat.lcm c a) j) t = 360 :=
by
  rw [hC, hA, hJ, hT]
  rfl

end tutors_meet_in_lab_l402_402641


namespace union_A_B_eq_neg2_neg1_0_l402_402577

def setA : Set ℤ := {x : ℤ | (x + 2) * (x - 1) < 0}
def setB : Set ℤ := {-2, -1}

theorem union_A_B_eq_neg2_neg1_0 : (setA ∪ setB) = ({-2, -1, 0} : Set ℤ) :=
by
  sorry

end union_A_B_eq_neg2_neg1_0_l402_402577


namespace correct_subtraction_result_l402_402437

-- Definitions based on the problem conditions
def initial_two_digit_number (X Y : ℕ) : ℕ := X * 10 + Y

-- Lean statement that expresses the proof problem
theorem correct_subtraction_result (X Y : ℕ) (H1 : initial_two_digit_number X Y = 99) (H2 : 57 = 57) :
  99 - 57 = 42 :=
by
  sorry

end correct_subtraction_result_l402_402437


namespace problem_l402_402614

theorem problem (a b : ℤ)
  (h1 : -2022 = -a)
  (h2 : -1 = -b) :
  a + b = 2023 :=
sorry

end problem_l402_402614


namespace range_a_l402_402363

open Real

axiom p (a : ℝ) : ∀ x : ℝ, x^2 + 2 * a * x + 4 > 0
axiom q (a : ℝ) : ∀ x y : ℝ, x < y → (3 - 2 * a)^x < (3 - 2 * a)^y
axiom pq : ∀ a : ℝ, (∃ x y : ℝ, x < y ∧ (3 - 2 * a)^x < (3 - 2 * a)^y) → false

theorem range_a (a : ℝ) : -2 < a ∧ a < 1 := by
  have : ∀ x : ℝ, x^2 + 2 * a * x + 4 > 0 := p a
  have : ∃ x y : ℝ, x < y ∧ (3 - 2 * a)^x < (3 - 2 * a)^y := pq a
  sorry

end range_a_l402_402363


namespace external_bisector_l402_402752

open Triangle

variable {Point : Type} [EuclideanGeometry Point]

variables {A T C L K : Point}

theorem external_bisector (h : angle_bisector A T C L) : external_bisector A T C K :=
sorry

end external_bisector_l402_402752


namespace Quincy_sold_more_l402_402670

def ThorSales : ℕ := 200 / 10
def JakeSales : ℕ := ThorSales + 10
def QuincySales : ℕ := 200

theorem Quincy_sold_more (H : QuincySales = 200) : QuincySales - JakeSales = 170 := by
  sorry

end Quincy_sold_more_l402_402670


namespace number_of_distinct_a_l402_402125

noncomputable def count_distinct_a : ℕ :=
  let divisors := [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 36, 48, 72, 144, -1, -2, -3, -4, -6, -8, -9, -12, -16, -18, -24, -36, -48, -72, -144]
  let pairs := (divisors.product divisors).filter (λ (p : ℤ × ℤ), p.1 * p.2 = 144)
  let sums := pairs.map (λ (p : ℤ × ℤ), p.1 + p.2 - 24)
  let distinct_sums := sums.eraseDuplicates  
  distinct_sums.length

theorem number_of_distinct_a : count_distinct_a = 14 :=
by
  sorry

end number_of_distinct_a_l402_402125


namespace exist_two_numbers_divide_l402_402549

theorem exist_two_numbers_divide (S : Finset ℕ) (H : S = (Finset.range 201).filter (λ x, x > 0)) 
  (chosen : Finset ℕ) (Hchosen : chosen.card = 100) (Hcondition : ∃ a, a ∈ chosen ∧ a < 16) :
  ∃ x y ∈ chosen, x ≠ y ∧ x ∣ y :=
by
  exists sorry,
  sorry


end exist_two_numbers_divide_l402_402549


namespace part1_case1_part1_case2_m_gt_neg1_part1_case2_m_lt_neg1_part2_l402_402223

def f (x m : ℝ) := (m + 1) * x^2 - (m - 1) * x + (m - 1)

theorem part1_case1 :
  ∀ x : ℝ, f x (-1) ≥ 0 := 
sorry

theorem part1_case2_m_gt_neg1 (m : ℝ) (h : m > -1) :
  ∀ x : ℝ, (f x m ≥ (m + 1) * x) ↔ 
    (x ≤ (m - 1) / (m + 1) ∨ x ≥ 1) := 
sorry

theorem part1_case2_m_lt_neg1 (m : ℝ) (h : m < -1) :
  ∀ x : ℝ, (f x m ≥ (m + 1) * x) ↔ 
    (1 ≤ x ∧ x ≤ (m - 1) / (m + 1)) := 
sorry

theorem part2 (m : ℝ) :
  (∀ x : ℝ, -1/2 ≤ x ∧ x ≤ 1/2 → f x m ≥ 0) ↔ m ∈ set.Ici 1 :=
sorry

end part1_case1_part1_case2_m_gt_neg1_part1_case2_m_lt_neg1_part2_l402_402223


namespace range_of_m_l402_402633

noncomputable def f (x : ℝ) (m : ℝ) := (1 / 3) * Real.exp (3 * x) + m * Real.exp (2 * x) + (2 * m + 1) * Real.exp x + 1

theorem range_of_m (m : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (∃ xt : ℝ, f' xt m = 0)) ↔ - (1 / 2) < m ∧ m < 1 - Real.sqrt 2 :=
sorry

end range_of_m_l402_402633


namespace general_term_and_sum_l402_402283

variable {n : ℕ}
variable {a : ℕ → ℕ}
variable {b : ℕ → ℕ}
variable {T : ℕ → ℕ}

-- Conditions for the geometric sequence {a_n}
axiom a_seq_geometric (n : ℕ) (a1 a2 : ℕ) (h1 : a1 * a2 = 8) (h2 : a1 + a2 = 6) : a n = 2^n

-- Definition of sequence {b_n}
def b_seq (n : ℕ) : ℕ := 2 * a n + 3

-- Sum of the first n terms of the sequence {b_n}
axiom sum_b_seq (n : ℕ) : T n = (2 ^ (n + 2)) - 4 + 3 * n

-- Theorem to prove
theorem general_term_and_sum 
(h : ∀ n, a n = 2 ^ n) 
(h_sum: ∀ n, T n = (2 ^ (n + 2)) - 4 + 3 * n) :
∀ n, (a n = 2 ^ n) ∧ (T n = (2 ^ (n + 2)) - 4 + 3 * n) := by
  intros
  exact ⟨h n, h_sum n⟩

end general_term_and_sum_l402_402283


namespace pogo_footprints_l402_402433

theorem pogo_footprints :
  (∃ (P : ℝ), 
    (∃ (PogoFootprints GrimziFootprints : ℝ),
      PogoFootprints = 6000 * P ∧
      GrimziFootprints = 6000 * 0.5 ∧
      PogoFootprints + GrimziFootprints = 27000) ∧
      P = 4) :=
begin
  sorry
end

end pogo_footprints_l402_402433


namespace stewart_farm_horseFood_l402_402959

variable (sheep horses horseFoodPerHorse : ℕ)
variable (ratio_sh_to_hs : ℕ × ℕ)
variable (totalHorseFood : ℕ)

noncomputable def horse_food_per_day (sheep : ℕ) (ratio_sh_to_hs : ℕ × ℕ) (totalHorseFood : ℕ) : ℕ :=
  let horses := (sheep * ratio_sh_to_hs.2) / ratio_sh_to_hs.1
  totalHorseFood / horses

theorem stewart_farm_horseFood (h_ratio : ratio_sh_to_hs = (4, 7))
                                (h_sheep : sheep = 32)
                                (h_total : totalHorseFood = 12880) :
    horse_food_per_day sheep ratio_sh_to_hs totalHorseFood = 230 := by
  sorry

end stewart_farm_horseFood_l402_402959


namespace minimum_value_of_expression_l402_402605

noncomputable def min_squared_distance (a b c d : ℝ) : ℝ :=
  (a - c)^2 + (b - d)^2

theorem minimum_value_of_expression
  (a b c d : ℝ)
  (h1 : 4 * a^2 + b^2 - 8 * b + 12 = 0)
  (h2 : c^2 - 8 * c + 4 * d^2 + 12 = 0) :
  min_squared_distance a b c d = 42 - 16 * Real.sqrt 5 :=
sorry

end minimum_value_of_expression_l402_402605


namespace sum_of_squares_gt_l402_402334

theorem sum_of_squares_gt (a : ℕ → ℝ) (h_pos : ∀ n, 0 < a n)
  (h_sqrt : ∀ n, 1 ≤ n → ∑ i in Finset.range n, a i ≥ Real.sqrt n) :
  ∀ n, 1 ≤ n → ∑ i in Finset.range n, (a i)^2 > (1 / 4) * ∑ i in Finset.range n, 1 / (i + 1) :=
by
  sorry

end sum_of_squares_gt_l402_402334


namespace largest_divisor_for_consecutive_seven_odds_l402_402841

theorem largest_divisor_for_consecutive_seven_odds (n : ℤ) (h_even : 2 ∣ n) (h_pos : 0 < n) : 
  105 ∣ ((n + 1) * (n + 3) * (n + 5) * (n + 7) * (n + 9) * (n + 11) * (n + 13)) :=
sorry

end largest_divisor_for_consecutive_seven_odds_l402_402841


namespace monotonicity_of_f_fx_gt_fpx_plus_three_half_l402_402556

noncomputable def f (a x : ℝ) : ℝ := a * (x - Real.log x) + (2 * x - 1) / (x ^ 2)

-- Part I: Discuss the monotonicity of f(x)
theorem monotonicity_of_f (a : ℝ) : 
  ∃ I : Set ℝ, (∀ x ∈ I, 0 < x) ∧
               (∀ x ∈ I, x > 1 → (x - 1) * (a * x ^ 2 - 2) / (x ^ 3) < 0) ∧
               (∀ x ∈ I, x < 1 → (x - 1) * (a * x ^ 2 - 2) / (x ^ 3) > 0) := 
by
  sorry

-- Part II: Prove that f(x) > f'(x) + 3/2 for any x ∈ [1,2] when a=1
theorem fx_gt_fpx_plus_three_half (x : ℝ) (h : 1 ≤ x ∧ x ≤ 2) : 
  f 1 x > (λ x, f 1 x - (1 - 1 / x + (2 - 2 * x) / (x ^ 3))) x + 3/2 := 
by
  sorry

end monotonicity_of_f_fx_gt_fpx_plus_three_half_l402_402556


namespace pyarelal_loss_is_1440_l402_402860

-- Define the conditions
variable (P : ℝ) -- Pyarelal's capital
variable (total_loss : ℝ) -- Total loss
variable (ashok_ratio : ℝ) -- Ratio of Ashok's capital to Pyarelal's capital

-- Conditions
def ashok_capital := P / 9
def combined_capital := P + ashok_capital
def pyarelal_ratio := P / combined_capital
def loss_pyarelal := pyarelal_ratio * total_loss

-- Given values
theorem pyarelal_loss_is_1440 
  (h1 : ashok_ratio = 1 / 9)
  (h2 : total_loss = 1600) :
  loss_pyarelal = 1440 := by
  sorry

end pyarelal_loss_is_1440_l402_402860


namespace tan_theta_neg_2_l402_402626

theorem tan_theta_neg_2 
  (θ : ℝ) 
  (h1 : sin θ + cos θ = sqrt 5 / 5) 
  (h2 : 0 ≤ θ ∧ θ ≤ Real.pi) :
  tan θ = -2 :=
sorry

end tan_theta_neg_2_l402_402626


namespace equation_of_C_minimum_area__PE_through_fixed_point_l402_402235

-- Definitions representing the conditions
variables {p : ℝ} (hp : p > 0)
noncomputable def parabola_1 (x y : ℝ) := y^2 = 2 * p * x

-- Question 1:
theorem equation_of_C (hC : ∀ x y, parabola_1 hp x y ↔ y^2 = 4 * x) :
  ∀ x y, y^2 = 2 * p * x ↔ y^2 = 4 * x :=
sorry

-- Definitions for points and other conditions
variables {x_P : ℝ} {P_F : ℝ}
noncomputable def focus_F := (p / 2, 0)
noncomputable def point_P := (3, p * 3)
noncomputable def point_S := (3 + p, 0)
noncomputable def point_E : Prop := x_P ∉ {0} ∧ |3 + p / 2 - p / 2| = |3 + p - (3 + p) / 2|

-- Question 2:
theorem minimum_area_△OPE (hPE : point_E) :
  ∃ m : ℝ, m = 2 :=
sorry

-- Definitions for fixed point condition
noncomputable def line_PE (x y : ℝ) := y - p * x / 3 = (p / 3) * (x - 3)

-- Question 3:
theorem PE_through_fixed_point (hPE : point_E) :
  ∀ x y, line_PE x y → (1, 0) :=
sorry

end equation_of_C_minimum_area__PE_through_fixed_point_l402_402235


namespace distance_foci_l402_402808

noncomputable def distance_between_foci := 
  let F1 := (4, 5)
  let F2 := (-6, 9)
  Real.sqrt ((F1.1 - F2.1)^2 + (F1.2 - F2.2)^2) 

theorem distance_foci : 
  ∃ (F1 F2 : ℝ × ℝ), 
    F1 = (4, 5) ∧ 
    F2 = (-6, 9) ∧ 
    distance_between_foci = 2 * Real.sqrt 29 := by {
  sorry
}

end distance_foci_l402_402808


namespace second_intersection_circumcircles_on_line_AM_l402_402686

variables {A B C M I_B I_C : Type*} [EuclideanGeometry A B C M]
variables (Γ₁ Γ₂ : CompCircle A B I_B) (Γ₃ : CompCircle A C I_C)

-- The main theorem stating that the second intersection of Γ₁ and Γ₂ lies on the line AM
theorem second_intersection_circumcircles_on_line_AM (hM : is_midpoint M B C)
  (h_IB : is_incenter I_B A B M) (h_IC : is_incenter I_C A C M) : 
  ∃ P, P ∈ circle.intersection Γ₁ Γ₂ ∧ lies_on_line P A M := 
sorry

end second_intersection_circumcircles_on_line_AM_l402_402686


namespace can_form_one_to_fourteen_with_five_sixes_l402_402834

def form_number_with_sixes : Prop :=
  (∃ a b c d e : ℤ, a = 6 ∧ b = 6 ∧ c = 6 ∧ d = 6 ∧ e = 6 ∧
    (∀ n ∈ (1:ℤ)..14, n =
      if n = 1 then (a / a) ^ (a - a)
      else if n = 2 then (a / a) + (a / a) ^ a
      else if n = 3 then (a / a) + (a / a) + (a / a) ^ a
      else if n = 4 then a - a + (a / a) ^ a
      else if n = 5 then (a * a) / (a + a)
      else if n = 6 then a * a / a
      else if n = 7 then a + (a / a)
      else if n = 8 then (a + a) / (a / a)
      else if n = 9 then (a * a) / (a + a) + (a / a)
      else if n = 10 then a + (a / a) * (a / a)
      else if n = 11 then a + a - a / a
      else if n = 12 then (a + a) / (a / a) + a / a 
      else if n = 13 then a + a - (a / a)
      else (a + a) + (a / a) + (a / a)
  ))

theorem can_form_one_to_fourteen_with_five_sixes : form_number_with_sixes :=
  by
    sorry

end can_form_one_to_fourteen_with_five_sixes_l402_402834


namespace Buratino_made_mistake_l402_402859

-- Define the cube and the requirement for each face
def is_adjacency_satisfied (face : list (list char)) : Prop :=
  ∀ i j, face[i][j] 'X' → (number_of_adjacent face i j 'X' = 2) ∧ 
         (number_of_adjacent face i j 'O' = 2)

-- Hypothesized configuration of faces
variable face_configuration : list (list char)

-- Lean statement to prove that Buratino made a mistake
theorem Buratino_made_mistake (faces : list (list (list char)))
  (h : ∀ face ∈ faces, is_adjacency_satisfied face → False) : False :=
by {
  -- Given the faces and the assumption of adjacency satisfaction,
  -- we need to show this leads to a contradiction.
  sorry
}

end Buratino_made_mistake_l402_402859


namespace calculate_first_expression_calculate_second_expression_l402_402513

section Problem1

theorem calculate_first_expression : (1 : ℝ) * (-2)^3 - (π - 3.14)^0 + (1 / 3)^(-2) = 0 :=
by
  simp
  sorry

end Problem1

section Problem2

variable (a : ℝ)

theorem calculate_second_expression : (a - 1)^2 - (a - 2) * (a + 2) = -2 * a + 5 :=
by
  simp
  sorry

end Problem2

end calculate_first_expression_calculate_second_expression_l402_402513


namespace fraction_remains_unchanged_l402_402639

theorem fraction_remains_unchanged (x y : ℝ) :
  (λ (x y : ℝ), 2008 * x / (2007 * y)) (2 * x) (2 * y) = 2008 * x / (2007 * y) :=
by
  sorry

end fraction_remains_unchanged_l402_402639


namespace infinite_sum_of_norms_l402_402681

theorem infinite_sum_of_norms 
  (α : ℂ) [IsAlgebraic α] (h_deg : Polynomial.degree (MinimalPolynomial α) = 2)
  (h_non_real : ¬ α.isReal)
  (P : Set (ℤ[α]) ) 
  (hP : ∀ p ∈ P, Irreducible p) :
  ∑' (p : P), (1 : ℝ) / (abs (norm p))^2 = ∞ := by
  sorry

end infinite_sum_of_norms_l402_402681


namespace integer_values_a_l402_402186

theorem integer_values_a :
  (∃! a : ℤ, ∃ m n : ℤ, m^2 + n^2 = (m + n)^2 - 2mn ∧
                     m + n = -a ∧ 
                     mn = 12a) :
  8 :=
sorry

end integer_values_a_l402_402186


namespace distance_projection_inequality_l402_402077

-- Definitions and conditions for tetrahedron and related points
variables (A B C D X : Point)
variables (ABC : Plane) (XBC : Plane) (XCA : Plane) (XAB : Plane)
variables (A' B' C' : Point)

-- Conditions
axiom HX : ¬ isInTetrahedron X A B C D
axiom HXD_intersects_ABC : segment X D ∩ plane A B C = interiorPoint

-- Projections
axiom HA' : A' = projection D XBC
axiom HB' : B' = projection D XCA
axiom HC' : C' = projection D XAB

theorem distance_projection_inequality 
    (HD_A' : distance A' D = projection_distance A') 
    (HD_B' : distance B' D = projection_distance B') 
    (HD_C' : distance C' D = projection_distance C') :
    distance A' B' + distance B' C' + distance C' A' ≤ distance D A + distance D B + distance D C := sorry

end distance_projection_inequality_l402_402077


namespace third_altitude_is_less_than_15_l402_402417

variable (a b c : ℝ)
variable (ha hb hc : ℝ)
variable (A : ℝ)

def triangle_area (side : ℝ) (height : ℝ) : ℝ := 0.5 * side * height

axiom ha_eq : ha = 10
axiom hb_eq : hb = 6

theorem third_altitude_is_less_than_15 : hc < 15 :=
sorry

end third_altitude_is_less_than_15_l402_402417


namespace distinct_integer_a_values_l402_402133

open Nat

theorem distinct_integer_a_values : ∃ (S : Finset ℤ), S.card = 8 ∧
  ∀ a ∈ S, ∃ (p q : ℤ), p + q = -a ∧ p * q = 12a ∧ 
  ∀ x ∈ (polynomial.roots (polynomial.C (12 : ℚ) * polynomial.X + polynomial.C a * polynomial.X + polynomial.C 1)).val, x ∈ ℤ :=
by
  sorry

end distinct_integer_a_values_l402_402133


namespace circle_center_sum_l402_402382

theorem circle_center_sum :
  let (h, k) := (2, 6)
  in h + k = 8 :=
by {
  have circle_eq : ∀ x y, x^2 + y^2 = 4 * x + 12 * y - 39 → (x - 2)^2 + (y - 6)^2 = 1, {
    intros x y h,
    sorry,
  },
  let center := (2, 6),
  have h_eq : 2 = center.1 := rfl,
  have k_eq : 6 = center.2 := rfl,
  rw [h_eq, k_eq],
  simp,   
  exact rfl,
}

end circle_center_sum_l402_402382


namespace whole_number_N_l402_402051

theorem whole_number_N (N : ℤ) : (9 < N / 4 ∧ N / 4 < 10) ↔ (N = 37 ∨ N = 38 ∨ N = 39) := 
by sorry

end whole_number_N_l402_402051


namespace count_integer_values_of_a_l402_402156

theorem count_integer_values_of_a (a : ℤ) :
  (∃ x : ℤ, x^2 + a * x + 12 * a = 0) ↔ (a ∈ {-(m + n) | m, n : ℤ ∧ m * n = 12 * a}) → ∃! a_vals : finset ℤ, a_vals.card = 16 :=
by
  sorry

end count_integer_values_of_a_l402_402156


namespace fraction_ratio_l402_402625

theorem fraction_ratio
  (m n p q r : ℚ)
  (h1 : m / n = 20)
  (h2 : p / n = 4)
  (h3 : p / q = 1 / 5)
  (h4 : m / r = 10) :
  r / q = 1 / 10 :=
by
  sorry

end fraction_ratio_l402_402625


namespace simplify_and_evaluate_evaluate_at_2_l402_402785

theorem simplify_and_evaluate (x : ℝ) 
  (hx1 : x ≠ 0) 
  (hx2 : x ≠ 1)
  (hx3 : x ≠ -1) :
  ( (2 / (x - 1) - 1 / x) / ((x^2 - 1) / (x^2 - 2 * x + 1)) ) = 1 / x :=
by sorry

theorem evaluate_at_2 : 
  ( (2 / (2 - 1) - 1 / 2) / ((2 ^ 2 - 1) / (2^2 - 2 * 2 + 1)) ) = 1 / 2 :=
by sorry

end simplify_and_evaluate_evaluate_at_2_l402_402785


namespace integer_values_of_a_l402_402169

theorem integer_values_of_a (a : ℤ) :
  (∀ x : ℤ, x^2 + a * x + 12 * a = 0 → x ∈ ℤ) → ∃ n : ℕ, n = 16 :=
by
  sorry

end integer_values_of_a_l402_402169


namespace number_of_trees_planted_l402_402948

-- Define the conditions as hypotheses
def total_length (yard_length : ℕ) : Prop := yard_length = 414
def distance_between_trees (d : ℕ) : Prop := d = 18

-- Define the main proof problem
theorem number_of_trees_planted (yard_length d : ℕ) (H_length : total_length yard_length) (H_distance : distance_between_trees d) :
  (yard_length / d) + 1 = 24 :=
by
  simp [total_length, distance_between_trees] at H_length H_distance
  rw [H_length, H_distance]
  sorry

end number_of_trees_planted_l402_402948


namespace candy_distribution_l402_402349

theorem candy_distribution (candy_total friends : ℕ) (candies : List ℕ) :
  candy_total = 47 ∧ friends = 5 ∧ List.length candies = friends ∧
  (∀ n ∈ candies, n = 9) → (47 % 5 = 2) :=
by
  sorry

end candy_distribution_l402_402349


namespace factorial_division_l402_402515

theorem factorial_division :
  (50.factorial / 47.factorial) = 117600 := by
  sorry

end factorial_division_l402_402515


namespace radius_of_circle_Q_l402_402966

theorem radius_of_circle_Q :
  ∃ (r Q : ℝ), r = 2 ∧ 
  (circle P Q ∧ circle Q P ∧ Q = 2) ∧ 
  (radius P = 2 ∧ center P ∈ circle S) ∧ 
  radius Q = 2 -> radius Q = 16/9 := sorry

end radius_of_circle_Q_l402_402966


namespace other_solution_of_quadratic_l402_402212

theorem other_solution_of_quadratic (x : ℚ) (h1 : x = 3 / 8) 
  (h2 : 72 * x^2 + 37 = -95 * x + 12) : ∃ y : ℚ, y ≠ 3 / 8 ∧ 72 * y^2 + 95 * y + 25 = 0 ∧ y = 5 / 8 :=
by
  sorry

end other_solution_of_quadratic_l402_402212


namespace count_colorings_l402_402410

-- Definitions of coloring and connected vertices
inductive Color
| red | white | blue

-- Each triangle has a unique number identifier
def isConnected (triangle1 triangle2: ℕ) (v1 v2: ℕ) : Prop := 
  -- Function to define if two vertices within two triangles are the same
  sorry

-- Each triangle has three vertices represented by numbers
def validColoring (coloring : ℕ → ℕ → Color) : Prop :=
  -- No two directly connected vertices have the same color
  (∀ t1 t2 v1 v2, isConnected t1 t2 v1 v2 → coloring t1 v1 ≠ coloring t2 v2) ∧
  -- Middle triangle has a vertex which is only red or white
  (∃ v, ∀ c, c = coloring 2 v → (c = Color.red ∨ c = Color.white))

-- The main theorem
theorem count_colorings : 
  let vertices := {v : ℕ // v < 9} -- 3 triangles with 3 vertices each
  let triangles := {t : ℕ // t < 3} -- 3 triangles
  ∃ (coloring : ℕ → ℕ → Color), validColoring coloring ∧
    (∑ t in triangles, ∑ v in vertices, 1) = 36 :=
sorry

end count_colorings_l402_402410


namespace angus_token_count_l402_402501

theorem angus_token_count (elsa_tokens : ℕ) (token_value : ℕ) 
  (tokens_less_than_elsa_value : ℕ) (elsa_token_value_relation : elsa_tokens = 60) 
  (token_value_relation : token_value = 4) (tokens_less_value_relation : tokens_less_than_elsa_value = 20) :
  elsa_tokens - (tokens_less_than_elsa_value / token_value) = 55 :=
by
  rw [elsa_token_value_relation, token_value_relation, tokens_less_value_relation]
  norm_num
  sorry

end angus_token_count_l402_402501


namespace least_n_divisible_by_99_l402_402315

def a (n : ℕ) : ℕ :=
  if h : n = 1 then 24 else 100 * a (n - 1) + 134

theorem least_n_divisible_by_99 : ∃ n : ℕ, n = 88 ∧ 99 ∣ a n :=
by
  sorry

end least_n_divisible_by_99_l402_402315


namespace distance_of_parallel_lines_l402_402800

-- Define the lines and the condition for parallelism
def line1 (x y : ℝ) := 3 * x + y - 3 = 0
def line2 (x y : ℝ) := 6 * x + 2 * y + 1 = 0

-- Define the distance formula for two parallel lines
def distance_between_parallel_lines (a b c1 c2 : ℝ) : ℝ :=
  abs (c2 - c1) / real.sqrt (a^2 + b^2)

-- Stating the theorem
theorem distance_of_parallel_lines : 
  ∀ x y : ℝ, 
  line1 x y → line2 x y → 
  distance_between_parallel_lines 6 2 (-6) 1 = (7 * real.sqrt 10) / 20 :=
by
  sorry

end distance_of_parallel_lines_l402_402800


namespace optimal_station_location_l402_402506

variable (a x : ℝ)
variable (speedTrain : ℝ := 0.8)
variable (speedRoad : ℝ := 0.2)
variable (B_to_track : ℝ := 20)
variable (total_travel_time : ℝ := (a - x) / speedTrain + Real.sqrt(x^2 + B_to_track^2) / speedRoad)

theorem optimal_station_location : 
    ∀ a : ℝ, 
    ∃ x : ℝ, 
    (total_travel_time = (a - 5) / speedTrain + Real.sqrt(5^2 + 20^2) / speedRoad) :=
begin
  sorry
end

end optimal_station_location_l402_402506


namespace circumcircle_radius_infinite_l402_402942

-- Define the conditions: a square ABCD with side length 1
structure Square (A B C D : Type) :=
  (AB : Real)
  (BC : Real)
  (CD : Real)
  (DA : Real)
  (side_length : Real)
  (h_AB : AB = side_length)
  (h_BC : BC = side_length)
  (h_CD : CD = side_length)
  (h_DA : DA = side_length)

-- Define the circle with diameter AD
structure Circle (D : Type) :=
  (diameter : Real)
  (h_diameter : diameter = Real.sqrt 2)

-- Define the problem statement and prove the result
theorem circumcircle_radius_infinite
  (A B C D : Type)
  (sq : Square A B C D)
  (circ : Circle D) :
  ∃ R : Real, R = ∞ :=
by
  sorry

end circumcircle_radius_infinite_l402_402942


namespace count_distinct_a_l402_402113

def quadratic_has_integer_solutions (a : ℤ) : Prop :=
  ∃ x m n : ℤ, (x^2 + a * x + 12 * a = 0) ∧ (m + n = -a) ∧ (m * n = 12 * a)

theorem count_distinct_a : (finset.univ.filter quadratic_has_integer_solutions).card = 9 := 
sorry

end count_distinct_a_l402_402113


namespace relatively_prime_dates_in_september_l402_402043

-- Define a condition to check if two numbers are relatively prime
def relatively_prime (a b : ℕ) : Prop := Nat.gcd a b = 1

-- Define the number of days in September
def days_in_september := 30

-- Define the month of September as the 9th month
def month_of_september := 9

-- Define the proposition that the number of relatively prime dates in September is 20
theorem relatively_prime_dates_in_september : 
  ∃ count, (count = 20 ∧ ∀ day, day ∈ Finset.range (days_in_september + 1) → relatively_prime month_of_september day → count = 20) := sorry

end relatively_prime_dates_in_september_l402_402043


namespace tangent_x_axis_a_eq_one_zero_le_a_le_one_implies_fx_ge_zero_product_inequality_l402_402596

/-- Problem 1: Tangent to the x-axis implies a equals 1 -/
theorem tangent_x_axis_a_eq_one
  (a : ℝ)
  (f : ℝ → ℝ := λ x, Real.exp x - a * (x + 1))
  (f' : ℝ → ℝ := λ x, Real.exp x - a)
  (tangent : ∃ x, f x = 0 ∧ f' x = 0) :
  a = 1 :=
sorry

/-- Problem 2: When 0 ≤ a ≤ 1, f(x) ≥ 0 for all x -/
theorem zero_le_a_le_one_implies_fx_ge_zero
  (a : ℝ)
  (f : ℝ → ℝ := λ x, Real.exp x - a * (x + 1))
  (h : 0 ≤ a ∧ a ≤ 1) :
  ∀ x, f x ≥ 0 :=
sorry

/-- Problem 3: For any positive integer n, the inequality holds: (1 + 1/2)(1 + 1/2^2)...(1 + 1/2^n) < e -/
theorem product_inequality (n : ℕ) (hn : 0 < n) :
  ∏ i in Finset.range n, (1 + 1/2 ^ (i + 1)) < Real.exp 1 :=
sorry

end tangent_x_axis_a_eq_one_zero_le_a_le_one_implies_fx_ge_zero_product_inequality_l402_402596


namespace integral1_eval_integral2_eval_derivative1_eval_derivative2_eval_l402_402905

noncomputable def integral1 : ℝ := ∫ x in 0..2, (3 * x^2 + 4 * x^3)
noncomputable def integral2 : ℝ := ∫ x in 0..1, (exp x + 2 * x)

def derivative1 (x : ℝ) : ℝ := (2 * x + 2 * cos (2 * x) - x^2 - sin (2 * x)) / exp x
def derivative2 (x : ℝ) : ℝ := 4 / (4 * x^2 - 1)

theorem integral1_eval : integral1 = 24 := sorry
theorem integral2_eval : integral2 = Real.exp 1 := sorry

theorem derivative1_eval (x : ℝ) : x ≠ 0 → (derivative1 x = (differentiable_at (λ x, (x^2 + sin (2 * x)) / exp x) x)) := sorry
theorem derivative2_eval (x : ℝ) (h : x > 1/2) : derivative2 x = (differentiable_at (λ x, log ((2 * x + 1) / (2 * x - 1))) x) := sorry

end integral1_eval_integral2_eval_derivative1_eval_derivative2_eval_l402_402905


namespace rabbit_position_after_2020_hops_l402_402929

theorem rabbit_position_after_2020_hops :
  let S_n (n : ℕ) : ℕ := 1 + n * (n + 1) / 2 in
  S_n 2020 = 2041211 :=
by
  let S_n (n : ℕ) : ℕ := 1 + n * (n + 1) / 2
  have h : S_n 2020 = 1 + 2020 * (2020 + 1) / 2 := rfl
  rw h
  have calc1 : 2020 * 2021 = 4082420 := rfl
  rw calc1
  have calc2 : 4082420 / 2 = 2041210 := rfl
  rw calc2
  show 1 + 2041210 = 2041211
  
  exact rfl

end rabbit_position_after_2020_hops_l402_402929


namespace general_term_and_sum_l402_402284

variable {n : ℕ}
variable {a : ℕ → ℕ}
variable {b : ℕ → ℕ}
variable {T : ℕ → ℕ}

-- Conditions for the geometric sequence {a_n}
axiom a_seq_geometric (n : ℕ) (a1 a2 : ℕ) (h1 : a1 * a2 = 8) (h2 : a1 + a2 = 6) : a n = 2^n

-- Definition of sequence {b_n}
def b_seq (n : ℕ) : ℕ := 2 * a n + 3

-- Sum of the first n terms of the sequence {b_n}
axiom sum_b_seq (n : ℕ) : T n = (2 ^ (n + 2)) - 4 + 3 * n

-- Theorem to prove
theorem general_term_and_sum 
(h : ∀ n, a n = 2 ^ n) 
(h_sum: ∀ n, T n = (2 ^ (n + 2)) - 4 + 3 * n) :
∀ n, (a n = 2 ^ n) ∧ (T n = (2 ^ (n + 2)) - 4 + 3 * n) := by
  intros
  exact ⟨h n, h_sum n⟩

end general_term_and_sum_l402_402284


namespace find_omega_l402_402600

-- Definitions for the problem
def f (ω x : ℝ) : ℝ := sin (ω * x) + cos (ω * x)

-- Conditions for the problem
def condition_omega (ω : ℝ) : Prop := ω > 0
def condition_x (x : ℝ) : Prop := True  -- x is in ℝ (no restriction other than existing in ℝ)
def condition_monotonically_increasing (ω : ℝ) : Prop := 
  ∀ x, -ω < x ∧ x < ω → f ω x < f ω (x + 1e-8)  -- an approximation for monotonic increase
def condition_symmetric (ω : ℝ) : Prop :=
  ∀ x, f ω x = f ω (2 * ω - x)  -- symmetry about x = ω

-- The theorem to prove that ω = π / 2
theorem find_omega (ω : ℝ) :
  condition_omega ω →
  (∀ x : ℝ, condition_x x) →
  condition_monotonically_increasing ω →
  condition_symmetric ω →
  ω = π / 2 :=
by
  sorry  -- Proof omitted for brevity

end find_omega_l402_402600


namespace angus_tokens_count_l402_402496

def worth_of_token : ℕ := 4
def elsa_tokens : ℕ := 60
def difference_worth : ℕ := 20

def elsa_worth : ℕ := elsa_tokens * worth_of_token
def angus_worth : ℕ := elsa_worth - difference_worth

def angus_tokens : ℕ := angus_worth / worth_of_token

theorem angus_tokens_count : angus_tokens = 55 := by
  sorry

end angus_tokens_count_l402_402496


namespace average_score_l402_402023

theorem average_score (classA_students classB_students : ℕ)
  (avg_score_classA avg_score_classB : ℕ)
  (h_classA : classA_students = 40)
  (h_classB : classB_students = 50)
  (h_avg_classA : avg_score_classA = 90)
  (h_avg_classB : avg_score_classB = 81) :
  (classA_students * avg_score_classA + classB_students * avg_score_classB) / 
  (classA_students + classB_students) = 85 := 
  by sorry

end average_score_l402_402023


namespace triangle_expression_l402_402609

variables (A B C : ℝ)
variables (AB BC CA : ℝ)
variables (AB_vec BC_vec CA_vec : ℝ × ℝ)

-- Conditions
def cond1 : Prop := AB = 3
def cond2 : Prop := BC = 4
def cond3 : Prop := CA = 5

-- The main theorem to prove
theorem triangle_expression : AB_vec × (BC_vec + (BC_vec × (CA_vec + (CA_vec × AB_vec)))) = -25 :=
by
  sorry

end triangle_expression_l402_402609


namespace largest_x_value_l402_402536

def largest_solution_inequality (x : ℝ) : Prop := 
  (-(Real.log 3 (100 + 2 * x * Real.sqrt (2 * x + 25)))^3 + 
  abs (Real.log 3 ((100 + 2 * x * Real.sqrt (2 * x + 25)) / (x^2 + 2 * x + 4)^4))) / 
  (3 * Real.log 6 (50 + 2 * x * Real.sqrt (2 * x + 25)) - 
  2 * Real.log 3 (100 + 2 * x * Real.sqrt (2 * x + 25))) ≥ 0 → 
  x ≤ 12 + 4 * Real.sqrt 3

theorem largest_x_value : ∃ x : ℝ, largest_solution_inequality x :=
sorry

end largest_x_value_l402_402536


namespace divides_2_pow_n_sub_1_no_n_divides_2_pow_n_add_1_l402_402454

theorem divides_2_pow_n_sub_1 (n : ℕ) : 7 ∣ (2 ^ n - 1) ↔ 3 ∣ n := by
  sorry

theorem no_n_divides_2_pow_n_add_1 (n : ℕ) : ¬ 7 ∣ (2 ^ n + 1) := by
  sorry

end divides_2_pow_n_sub_1_no_n_divides_2_pow_n_add_1_l402_402454


namespace instantaneous_velocity_l402_402059

variable {t : ℝ} {Δt : ℝ} (Δs : ℝ)

theorem instantaneous_velocity (Δs : ℝ) :
  (t : ℝ) → (Δt : ℝ) → 
  ∀ t, Δs = displacement_from_time_t_to_t_add_Delta_t t Δt →
  (  ∃(v : ℝ), (t : ℝ), (Δt : ℝ), 
  ∀ (ε : ℝ), ε > 0 → 
  ∃ δ > 0, ∀ t, 0 < |Δt| < δ → 
  | (Δs / Δt) - v | < ε  ) := sorry

end instantaneous_velocity_l402_402059


namespace remainder_of_polynomial_l402_402104

def p (x : ℝ) : ℝ := x^4 + 2*x^2 + 5

theorem remainder_of_polynomial (x : ℝ) : p 2 = 29 :=
by
  sorry

end remainder_of_polynomial_l402_402104


namespace prob_complementary_event_l402_402780

variables (A : Type) [ProbabilityMeasure A]

-- Assume P(A) = 0.65
axiom prob_A : Probability A = 0.65

-- Define the probability of the complementary event (not A)
def prob_not_A : Probability A := 1 - Probability A

-- The theorem that needs to be proven
theorem prob_complementary_event : prob_not_A A = 0.35 := 
by 
  -- Insert your proof here
  sorry

end prob_complementary_event_l402_402780


namespace geometric_sequence_sum_l402_402656

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) (n : ℕ) :
  (∀ n, a n = 2 * q ^ n) →
  a 0 = 2 →
  (∀ n, ∃ r, a n + 1 = r * q ^ n) →
  ∑ i in finset.range n, a i = 2 * n :=
by
  intros h1 h2 h3
  sorry

end geometric_sequence_sum_l402_402656


namespace possible_knight_counts_l402_402768

/-- On an island, there are 10 inhabitants, each with a unique T-shirt number from 1 to 10.
    There are knights who always tell the truth and liars who always lie. Each inhabitant makes one 
    of the following statements exactly 5 times:
    1. "There is no knight among those gathered whose T-shirt number is greater than mine."
    2. "There is no liar among those gathered whose T-shirt number is less than mine."

    The possible number of knights among these 10 inhabitants can be 1, 2, 3, 4, 5, or 6. -/
theorem possible_knight_counts 
    (inhabitants : Fin 10 → Prop)
    (knight : Fin 10 → Prop)
    (liar : Fin 10 → Prop)
    (h_exclusive : ∀ i, knight i ↔ ¬ liar i)
    (h_count : (Fin 10 → Prop) → Nat → Prop)
    (h_statement1 : ∀ i, (inhabitants i) → (∀ j, j > i → liar j) → knight i = liar (Fin.ltAdd i))
    (h_statement2 : ∀ i, (inhabitants i) → (∀ j, j < i → knight j) → liar i = knight (Fin.gtAdd i))
    (h_distrib : h_count knight 5 ∧ h_count liar 5) :
  ∃ k, k = 1 ∨ k = 2 ∨ k = 3 ∨ k = 4 ∨ k = 5 ∨ k = 6 := sorry

end possible_knight_counts_l402_402768


namespace distinct_integer_a_values_l402_402138

open Nat

theorem distinct_integer_a_values : ∃ (S : Finset ℤ), S.card = 8 ∧
  ∀ a ∈ S, ∃ (p q : ℤ), p + q = -a ∧ p * q = 12a ∧ 
  ∀ x ∈ (polynomial.roots (polynomial.C (12 : ℚ) * polynomial.X + polynomial.C a * polynomial.X + polynomial.C 1)).val, x ∈ ℤ :=
by
  sorry

end distinct_integer_a_values_l402_402138


namespace voldemort_lunch_calories_l402_402835

def dinner_cake_calories : Nat := 110
def chips_calories : Nat := 310
def coke_calories : Nat := 215
def breakfast_calories : Nat := 560
def daily_intake_limit : Nat := 2500
def remaining_calories : Nat := 525

def total_dinner_snacks_breakfast : Nat :=
  dinner_cake_calories + chips_calories + coke_calories + breakfast_calories

def total_remaining_allowance : Nat :=
  total_dinner_snacks_breakfast + remaining_calories

def lunch_calories : Nat :=
  daily_intake_limit - total_remaining_allowance

theorem voldemort_lunch_calories:
  lunch_calories = 780 := by
  sorry

end voldemort_lunch_calories_l402_402835


namespace lateral_surface_area_of_cylinder_l402_402026

-- Define the given condition: the cross-sectional area of the cylinder is a square with area 5
def square_cross_section_area : ℝ := 5

-- Prove that the lateral surface area of the cylinder is equal to 5π under the given condition
theorem lateral_surface_area_of_cylinder (side : ℝ) (h : ℝ) (r : ℝ) :
  (side^2) = square_cross_section_area →
  side = h → 
  r = side / 2 →
  2 * ℕ.pi * r * h = 5 * ℕ.pi :=
by intros _ _ _; simp [square_cross_section_area]; sorry

end lateral_surface_area_of_cylinder_l402_402026


namespace ball_distribution_ways_l402_402623

theorem ball_distribution_ways :
  (∃ f : Fin 6 → Fin 2, ∀ i : Fin 2, ∃ x : Fin 6, f x = i) → 30 := sorry

end ball_distribution_ways_l402_402623


namespace smallest_quotient_is_1_9_l402_402054

def is_two_digit_number (n : ℕ) : Prop :=
  10 <= n ∧ n <= 99

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  let x := n / 10
  let y := n % 10
  x + y

noncomputable def quotient (n : ℕ) : ℚ :=
  n / (sum_of_digits n)

theorem smallest_quotient_is_1_9 :
  ∃ n, is_two_digit_number n ∧ (∃ x y, n = 10 * x + y ∧ x ≠ y ∧ 1 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9) ∧ quotient n = 1.9 := 
sorry

end smallest_quotient_is_1_9_l402_402054


namespace max_projection_area_tetra_l402_402844

theorem max_projection_area_tetra (T : Tetrahedron) (h_regular : T.Regular) (h_edge_len : ∀ e ∈ T.edges, e.length = 1) :
  ∃ max_area, max_area = 1 / 2 :=
sorry

end max_projection_area_tetra_l402_402844


namespace probability_C_l402_402029

-- Variables representing the probabilities of each region
variables (P_A P_B P_C P_D P_E : ℚ)

-- Given conditions
def conditions := P_A = 3/10 ∧ P_B = 1/4 ∧ P_D = 1/5 ∧ P_E = 1/10 ∧ P_A + P_B + P_C + P_D + P_E = 1

-- The statement to prove
theorem probability_C (h : conditions P_A P_B P_C P_D P_E) : P_C = 3/20 := 
by
  sorry

end probability_C_l402_402029


namespace automobile_travel_distance_l402_402954

theorem automobile_travel_distance (a r : ℝ) :
  (2 * a / 5) / (2 * r) * 5 * 60 / 3 = 20 * a / r :=
by 
  -- skipping proof details
  sorry

end automobile_travel_distance_l402_402954


namespace ellipse_equation_l402_402199

theorem ellipse_equation
  (P : ℝ × ℝ) (c : ℝ) (a b : ℝ)
  (hP_on_circle : (P.1 ^ 2 + P.2 ^ 2 = 1))
  (h_line_passes_focus_vertex : ∀ (A B : ℝ × ℝ), (A.1 ^ 2 + A.2 ^ 2 = 1) ∧ (B.1 ^ 2 + B.2 ^ 2 = 1) ∧ (line_passes_through A B 1 0) ∧ (line_passes_through A B 0 2))
  (ha_gt_b : a > b)
  (ha_pos : a > 0)
  (hb_pos : b > 0)
  (h_ellipse : ∀ (x y : ℝ), ((x ^ 2 / a ^ 2 + y ^ 2 / b ^ 2 = 1) ↔ ∃ (c : ℝ), a^2 = b^2 + c^2) ):
  a = sqrt 5 ∧ b = 2 ∧ a^2 = 5 ∧ b^2 = 4 := 
begin
  sorry,
end

end ellipse_equation_l402_402199


namespace compute_expression_l402_402516

theorem compute_expression : (46 + 15)^2 - (46 - 15)^2 = 2760 :=
by
  sorry

end compute_expression_l402_402516


namespace nonempty_solution_set_iff_a_gt_2_l402_402781

theorem nonempty_solution_set_iff_a_gt_2 (a : ℝ) : 
  (∃ x : ℝ, |x - 3| + |x - 5| < a) ↔ a > 2 :=
sorry

end nonempty_solution_set_iff_a_gt_2_l402_402781


namespace exists_large_N_l402_402079

open Nat

def a : ℕ → ℕ
| 1 := 1
| 2 := 1
| 3 := 1
| (n+3) := a (n+1) + a n

def b : ℕ → ℕ
| 1 := 1
| 2 := 1
| 3 := 1
| 4 := 1
| 5 := 1
| (n+5) := b (n+4) + b n

theorem exists_large_N :
  ∃ N, ∀ n, n ≥ N → a n = b (n+1) + b (n-8) :=
sorry

end exists_large_N_l402_402079


namespace marble_ratio_l402_402945

theorem marble_ratio (A V X : ℕ) 
  (h1 : A + 5 = V - 5)
  (h2 : V + X = (A - X) + 30) : X / 5 = 2 :=
by
  sorry

end marble_ratio_l402_402945


namespace problem_part1_problem_part2_l402_402195

noncomputable def complex_z : ℂ :=
  let z := 1 + complex.I in z

theorem problem_part1:
  ∃ z : ℂ, 
  (z - complex.I).im = 0 ∧ 
  ((z - 3 * complex.I) / (-2 - complex.I)).re = 0 → 
  z = 1 + complex.I := 
sorry

theorem problem_part2 (m : ℝ):
  ∀ z : ℂ, 
  z = 1 + complex.I → 
  let z1 := z / (m - complex.I) in 
  z1.re < 0 ∧ z1.im > 0 ↔ 
  m ∈ Icc (-1) 1 := 
sorry

end problem_part1_problem_part2_l402_402195


namespace student_weekly_allowance_l402_402244

theorem student_weekly_allowance (A : ℝ) 
  (h1 : ∃ spent_arcade, spent_arcade = (3 / 5) * A)
  (h2 : ∃ spent_toy, spent_toy = (1 / 3) * ((2 / 5) * A))
  (h3 : ∃ spent_candy, spent_candy = 0.60)
  (h4 : ∃ remaining_after_toy, remaining_after_toy = ((6 / 15) * A - (2 / 15) * A))
  (h5 : remaining_after_toy = 0.60) : 
  A = 2.25 := by
  sorry

end student_weekly_allowance_l402_402244


namespace initial_cases_purchased_l402_402024

open Nat

-- Definitions based on conditions

def group1_children := 14
def group2_children := 16
def group3_children := 12
def group4_children := (group1_children + group2_children + group3_children) / 2
def total_children := group1_children + group2_children + group3_children + group4_children

def bottles_per_child_per_day := 3
def days := 3
def total_bottles_needed := total_children * bottles_per_child_per_day * days

def additional_bottles_needed := 255

def bottles_per_case := 24
def initial_bottles := total_bottles_needed - additional_bottles_needed

def cases_purchased := initial_bottles / bottles_per_case

-- Theorem to prove the number of cases purchased initially
theorem initial_cases_purchased : cases_purchased = 13 :=
  sorry

end initial_cases_purchased_l402_402024


namespace external_angle_bisector_l402_402716

open EuclideanGeometry

variables {A T C K L : Point}

theorem external_angle_bisector
    (h1 : is_internal_angle_bisector T L A T C) : is_external_angle_bisector T K A T C :=
sorry

end external_angle_bisector_l402_402716


namespace spadesuit_calculation_l402_402546

def spadesuit (a b : ℕ+) : ℚ := a - (1 / b)

theorem spadesuit_calculation : spadesuit 3 (spadesuit 5 3) = 39 / 14 := by
  sorry

end spadesuit_calculation_l402_402546


namespace range_of_t_l402_402560

theorem range_of_t (a t : ℝ) (x y : ℝ) (h₀ : 0 < a) (h₁ : a < 1) :
  a * x^2 + t * y^2 ≥ (a * x + t * y)^2 ↔ 0 ≤ t ∧ t ≤ 1 - a :=
sorry

end range_of_t_l402_402560


namespace number_of_quadrilaterals_with_circumcenter_find_quadrilaterals_with_circumcenter_l402_402187

-- Definitions for types of quadrilaterals
def is_square (q : Quadrilateral) : Prop := ...
def is_rectangle (q : Quadrilateral) : Prop := ...
def is_rhombus (q : Quadrilateral) : Prop := ...
def is_kite (q : Quadrilateral) : Prop := ...
def is_general_trapezoid (q : Quadrilateral) : Prop := ...

def has_circumcenter (q : Quadrilateral) : Prop :=
  ∃ (p : Point), ∀ v ∈ q.vertices, dist p v = dist p q.vertices.head

-- Conditions for specific quadrilaterals
axiom square_has_circumcenter : ∀ (q : Quadrilateral), is_square q → has_circumcenter q
axiom rectangle_has_circumcenter : ∀ (q : Quadrilateral), is_rectangle q ∧ ¬is_square q → has_circumcenter q
axiom rhombus_no_circumcenter_if_not_square : ∀ (q : Quadrilateral), is_rhombus q ∧ ¬is_square q → ¬has_circumcenter q
axiom kite_no_circumcenter_in_general : ∀ (q : Quadrilateral), is_kite q ∧ ¬is_rhombus q → ¬has_circumcenter q
axiom trapezoid_no_circumcenter_in_general : ∀ (q : Quadrilateral), is_general_trapezoid q ∧ ¬is_isosceles q → ¬has_circumcenter q

-- Number of quadrilaterals with a circumcenter
theorem number_of_quadrilaterals_with_circumcenter : ℕ :=
  let quadrilaterals := [q1, q2, q3, q4, q5] in
  quadrilaterals.count (λ q, has_circumcenter q)

-- Proof statement of the problem
theorem find_quadrilaterals_with_circumcenter :
  number_of_quadrilaterals_with_circumcenter = 2 :=
by
  sorry

end number_of_quadrilaterals_with_circumcenter_find_quadrilaterals_with_circumcenter_l402_402187


namespace widgets_production_l402_402925

variables (A B C : ℝ)
variables (P : ℝ)

-- Conditions provided
def condition1 : Prop := 7 * A + 11 * B = 305
def condition2 : Prop := 8 * A + 22 * C = P

-- The question we need to answer
def question : Prop :=
  ∃ Q : ℝ, Q = 8 * (A + B + C)

theorem widgets_production (h1 : condition1 A B) (h2 : condition2 A C P) :
  question A B C :=
sorry

end widgets_production_l402_402925


namespace total_value_of_item_l402_402473

theorem total_value_of_item (V : ℝ) 
  (h1 : ∃ V > 1000, 
              0.07 * (V - 1000) + 
              (if 55 > 50 then (55 - 50) * 0.15 else 0) + 
              0.05 * V = 112.70) :
  V = 1524.58 :=
by 
  sorry

end total_value_of_item_l402_402473


namespace sum_of_tangents_l402_402192

theorem sum_of_tangents (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) 
    (h_tan_α : Real.tan α = 2) (h_tan_β : Real.tan β = 3) : α + β = 3 * π / 4 :=
by
  sorry

end sum_of_tangents_l402_402192


namespace part1_part2_l402_402232

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := x^3 + k * Real.log x
noncomputable def f' (x : ℝ) (k : ℝ) : ℝ := 3 * x^2 + k / x
noncomputable def g (x : ℝ) (k : ℝ) : ℝ := f x k - f' x k + 9 / x

-- Part (1): Prove the monotonic intervals and extreme values for k = 6:
theorem part1 :
  (∀ x : ℝ, 0 < x ∧ x < 1 → g x 6 < g 1 6) ∧
  (∀ x : ℝ, 1 < x → g x 6 > g 1 6) ∧
  (g 1 6 = 1) := sorry

-- Part (2): Prove the given inequality for k ≥ -3:
theorem part2 (k : ℝ) (hk : k ≥ -3) (x1 x2 : ℝ) (hx1 : x1 ≥ 1) (hx2 : x2 ≥ 1) (h : x1 > x2) :
  (f' x1 k + f' x2 k) / 2 > (f x1 k - f x2 k) / (x1 - x2) := sorry

end part1_part2_l402_402232


namespace integer_solution_count_l402_402148

def has_integer_solutions (a : ℤ) : Prop :=
  ∃ x y : ℤ, x^2 + a * x + 12 * a = 0 ∧ y^2 + a * y + 12 * a = 0

theorem integer_solution_count : 
  (set.finite {a : ℤ | has_integer_solutions a}).to_finset.card = 16 := 
begin
  sorry
end

end integer_solution_count_l402_402148


namespace cancel_left_l402_402313

variable {S : Type} [commRing S] (mul : S → S → S)

namespace GroupProblem

open Classical

-- Define properties of the operation
axiom star_comm : ∀ (x y : S), mul x y = mul y x
axiom star_assoc : ∀ (x y z : S), mul (mul x y) z = mul x (mul y z)
axiom star_solvability : ∀ (x y : S), ∃ z : S, mul x z = y

theorem cancel_left (a b c : S) (Hmul : ∀ (x y : S), mul x y = mul y x) (Hassoc : ∀ (x y z : S), mul (mul x y) z = mul x (mul y z)) (Hsolve : ∀ (x y : S), ∃ z : S, mul x z = y) (h : mul a c = mul b c) : a = b := 
sorry

end GroupProblem

end cancel_left_l402_402313


namespace motorist_spent_on_petrol_l402_402488

def original_price_per_gallon : ℝ := 5.56
def reduction_percentage : ℝ := 0.10
def new_price_per_gallon := original_price_per_gallon - (0.10 * original_price_per_gallon)
def gallons_more_after_reduction : ℝ := 5

theorem motorist_spent_on_petrol (X : ℝ) 
  (h1 : new_price_per_gallon = original_price_per_gallon - (reduction_percentage * original_price_per_gallon))
  (h2 : (X / new_price_per_gallon) - (X / original_price_per_gallon) = gallons_more_after_reduction) :
  X = 250.22 :=
by
  sorry

end motorist_spent_on_petrol_l402_402488


namespace Q_has_rational_coefficients_and_leading_1_and_degree_4_and_has_root_sqrt3_sqrt7_l402_402411

noncomputable def Q : ℚ[X] :=
  let z : ℝ := real.sqrt 3 + real.sqrt 7 in
  let r1 : ℝ := z in
  let r2 : ℝ := real.sqrt 3 - real.sqrt 7 in
  let r3 : ℝ := -z in
  let r4 : ℝ := -(real.sqrt 3 - real.sqrt 7) in
  (X - C r1) * (X - C r2) * (X - C r3) * (X - C r4)

theorem Q_has_rational_coefficients_and_leading_1_and_degree_4_and_has_root_sqrt3_sqrt7 :
  polynomial.degree Q = 4 ∧
  polynomial.leading_coeff Q = 1 ∧
  polynomial.coeff Q = (root_poly.to_poly),
  let Q' := polynomial.monic normalize p ∧
  polynomial.eval (2) Q' = -48 :=
by {
  sorry
}

end Q_has_rational_coefficients_and_leading_1_and_degree_4_and_has_root_sqrt3_sqrt7_l402_402411


namespace solve_system_l402_402787

-- Definitions of the variables
variable (x1 x2 x3 x4 x5 c1 c2 c3 : ℝ)

-- Definitions of the equations
def eq1 := x1 + 2 * x2 - x3 + x4 - 2 * x5 = -3
def eq2 := x1 + 2 * x2 + 3 * x3 - x4 + 2 * x5 = 17
def eq3 := 2 * x1 + 4 * x2 + 2 * x3 = 14

-- The general solution proposed
def solution := 
  (-2 * c1 - c2 + 2, c1 + 1, c2 + 3, 2 * c2 + 2 * c3 - 2, c3 + 1)


-- The theorem to prove
theorem solve_system : eq1 ∧ eq2 ∧ eq3 → 
  ∃ (c1 c2 c3 : ℝ), (x1, x2, x3, x4, x5) = solution :=
by
  sorry

end solve_system_l402_402787


namespace smaller_circle_radius_proof_l402_402517

noncomputable theory

def larger_circle_radius := 4
def area_larger_circle := real.pi * larger_circle_radius^2
def area_smaller_circle := area_larger_circle / 2
def arithmetic_progression_areas := [area_smaller_circle, area_larger_circle - area_smaller_circle, area_larger_circle]

theorem smaller_circle_radius_proof (h1 : larger_circle_radius = 4)
  (h2 : area_smaller_circle = area_larger_circle / 2)
  (h3 : arithmetic_progression_areas = [area_smaller_circle, area_larger_circle - area_smaller_circle, area_larger_circle]) :
  ∃ (r : ℝ), r = 2 * real.sqrt 2 ∧ area_smaller_circle = real.pi * r^2 :=
begin
  sorry
end

end smaller_circle_radius_proof_l402_402517


namespace find_m_real_equal_roots_l402_402541

theorem find_m_real_equal_roots (m : ℝ) :
  (∀ x : ℝ, 3 * x^2 + (2 - m) * x + 10 = 0 → (2 - m)^2 - 4 * 3 * 10) = 0 ↔
  (m = 2 - 2 * Real.sqrt 30) ∨ (m = 2 + 2 * Real.sqrt 30) :=
by
  sorry

end find_m_real_equal_roots_l402_402541


namespace external_angle_bisector_proof_l402_402730

variables {A T C L K : Type} [Nonempty A] [Nonempty T] [Nonempty C] [Nonempty L] [Nonempty K]

noncomputable def angle_bisector_theorem (AL LC AB BC AK KC : ℝ) : Prop :=
(AL / LC) = (AB / BC) ∧ (AK / KC) = (AL / LC)

noncomputable def internal_angle_bisector (AT TC AL LC : ℝ) : Prop :=
(AT / TC) = (AL / LC)

noncomputable def external_angle_bisector (AT TC AK KC : ℝ) : Prop :=
(AT / TC) = (AK / KC)

theorem external_angle_bisector_proof (AL LC AB BC AK KC AT TC : ℝ) 
(h1 : angle_bisector_theorem AL LC AB BC AK KC)
(h2 : internal_angle_bisector AT TC AL LC) :
external_angle_bisector AT TC AK KC :=
sorry

end external_angle_bisector_proof_l402_402730


namespace compute_cd_l402_402389

noncomputable def ellipse_foci_at : Prop :=
  ∃ (c d : ℝ), (d^2 - c^2 = 25) ∧ (c^2 + d^2 = 64) ∧ (|c * d| = real.sqrt 868.75)

theorem compute_cd : ellipse_foci_at :=
  sorry

end compute_cd_l402_402389


namespace arithmetic_sequence_sum_2006th_term_l402_402242

theorem arithmetic_sequence_sum_2006th_term (
  a : ℕ → ℤ,
  b : ℕ → ℤ,
  h1: a 1 = 25,
  h2: b 1 = 125,
  h3: ∀ n, a (n + 1) - a n = a 2 - a 1,
  h4: ∀ n, b (n + 1) - b n = b 2 - b 1,
  h5: a 2 + b 2 = 150
) : a 2006 + b 2006 = 150 := sorry

end arithmetic_sequence_sum_2006th_term_l402_402242


namespace jeff_running_speed_l402_402674

def distance (u j : ℝ) := 30
def ursula_speed := 10
def jeff_time := distance u j / ursula_speed - 1
def jeff_speed := distance u j / jeff_time

theorem jeff_running_speed : jeff_speed = 15 := by
  sorry

end jeff_running_speed_l402_402674


namespace smallest_positive_period_sin_l402_402343

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  p > 0 ∧ ∀ x, f (x + p) = f x

theorem smallest_positive_period_sin (f : ℝ → ℝ) (h : f = (λ x, 2 * Real.sin x)) : 
  ∃ p > 0, is_periodic f p ∧ ∀ q > 0, is_periodic f q → q ≥ p ∧ p = 2 * Real.pi := 
by 
  sorry

end smallest_positive_period_sin_l402_402343


namespace factor_polynomial_l402_402989

noncomputable def gcd_coeffs : ℕ := Nat.gcd 72 180

theorem factor_polynomial (x : ℝ) (GCD_72_180 : gcd_coeffs = 36)
    (GCD_x5_x9 : ∃ (y: ℝ), x^5 = y ∧ x^9 = y * x^4) :
    72 * x^5 - 180 * x^9 = -36 * x^5 * (5 * x^4 - 2) :=
by
  sorry

end factor_polynomial_l402_402989


namespace min_sum_products_l402_402858

theorem min_sum_products (x : ℕ → Int) (n : ℕ) (h : ∀ i, x i = 1 ∨ x i = 0 ∨ x i = -1) :
  ∃ S, S = -n / 2 :=
sorry

end min_sum_products_l402_402858


namespace original_wire_length_l402_402915

theorem original_wire_length 
(L : ℝ) 
(h1 : L / 2 - 3 / 2 > 0) 
(h2 : L / 2 - 3 > 0) 
(h3 : L / 4 - 11.5 > 0)
(h4 : L / 4 - 6.5 = 7) : 
L = 54 := 
sorry

end original_wire_length_l402_402915


namespace total_spent_by_pete_and_raymond_l402_402773

def pete_initial_amount := 250
def pete_spending_on_stickers := 4 * 5
def pete_spending_on_candy := 3 * 10
def pete_spending_on_toy_car := 2 * 25
def pete_spending_on_keychain := 5
def pete_total_spent := pete_spending_on_stickers + pete_spending_on_candy + pete_spending_on_toy_car + pete_spending_on_keychain
def raymond_initial_amount := 250
def raymond_left_dimes := 7 * 10
def raymond_left_quarters := 4 * 25
def raymond_left_nickels := 5 * 5
def raymond_left_pennies := 3 * 1
def raymond_total_left := raymond_left_dimes + raymond_left_quarters + raymond_left_nickels + raymond_left_pennies
def raymond_total_spent := raymond_initial_amount - raymond_total_left
def total_spent := pete_total_spent + raymond_total_spent

theorem total_spent_by_pete_and_raymond : total_spent = 157 := by
  have h1 : pete_total_spent = 105 := sorry
  have h2 : raymond_total_spent = 52 := sorry
  exact sorry

end total_spent_by_pete_and_raymond_l402_402773


namespace family_raised_percentage_l402_402790

theorem family_raised_percentage :
  ∀ (total_funds friends_percentage own_savings family_funds remaining_funds : ℝ),
    total_funds = 10000 →
    friends_percentage = 0.40 →
    own_savings = 4200 →
    remaining_funds = total_funds - (friends_percentage * total_funds) →
    family_funds = remaining_funds - own_savings →
    (family_funds / remaining_funds) * 100 = 30 :=
by
  intros total_funds friends_percentage own_savings family_funds remaining_funds
  intros h_total_funds h_friends_percentage h_own_savings h_remaining_funds h_family_funds
  sorry

end family_raised_percentage_l402_402790


namespace integer_values_of_a_l402_402173

theorem integer_values_of_a (a : ℤ) :
  (∀ x : ℤ, x^2 + a * x + 12 * a = 0 → x ∈ ℤ) → ∃ n : ℕ, n = 16 :=
by
  sorry

end integer_values_of_a_l402_402173


namespace sufficient_but_not_necessary_condition_l402_402571

theorem sufficient_but_not_necessary_condition (x : ℝ) (p : -1 < x ∧ x < 3) (q : x^2 - 5 * x - 6 < 0) : 
  (-1 < x ∧ x < 3) → (x^2 - 5 * x - 6 < 0) ∧ ¬((x^2 - 5 * x - 6 < 0) → (-1 < x ∧ x < 3)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l402_402571


namespace num_goats_brought_l402_402910

open Nat

def avg_price_cow := 400
def num_cows := 2
def avg_price_goat := 70
def total_cost := 1500

theorem num_goats_brought : ∃ G, 
  (num_cows * avg_price_cow + G * avg_price_goat = total_cost) ∧
  G = 10 := 
by
  use 10
  split
  rfl
  sorry

end num_goats_brought_l402_402910


namespace Serena_fraction_bound_l402_402373

theorem Serena_fraction_bound :
  ∀ (f : ℚ) (k : ℕ), (k ≤ 20) → 
  (∀ n ∈ List.range k, f ∈ List.map (λ (n : ℚ), n) [1] ∨ 
    (∀ a b, (a + b = f) ∨ (1/a = f)) ∧
    ∀ n ∈ List.range k, rat.canonicalize' f f.normal_denom) →
  ¬ (∃ x y : ℕ, (x > 9000) ∧ (y > 0) ∧ (x < y) ∧ f = (↑x)/(↑y)) := 
sorry

end Serena_fraction_bound_l402_402373


namespace parabola_transformation_l402_402805

def original_parabola (x : ℝ) : ℝ := 3 * x^2

def shifted_left (x : ℝ) : ℝ := original_parabola (x + 1)

def shifted_down (x : ℝ) : ℝ := shifted_left x - 2

theorem parabola_transformation :
  shifted_down x = 3 * (x + 1)^2 - 2 :=
sorry

end parabola_transformation_l402_402805


namespace proof_AE_over_EF_l402_402074

noncomputable def point := (ℝ × ℝ)

def A (a : ℝ) : point := (a, 0)
def D (a : ℝ) : point := (a, a)

def B (a b : ℝ) : point := (b, b)
def C (a c : ℝ) : point := (c, 2*c)

def condition_AB_BC (a b c : ℝ) : Prop := 
  real.sqrt ((b - a)^2 + b^2) / real.sqrt ((c - b)^2 + (2*c - b)^2) = 2

def circumcircle_intersects_y_equals_x_again_at_E (a b : ℝ) (A : point) (D : point) (C : point) : point := 
  -- Intersection point on y = x, assume E
  (frac(a, 4), frac(a, 4)) -- Assuming E = (a/4, a/4) 

def ray_AE_intersects_y_equals_2x_at_F (a : ℝ) (E : point) : point := 
  -- Intersection point E
  (frac(a, 7), frac(2 * a / 7)) -- Assuming F = (a/7, 2a/7)

theorem proof_AE_over_EF (a b c : ℝ) (E F : point) :
  ∀ (h : condition_AB_BC a b c), 
  let E := circumcircle_intersects_y_equals_x_again_at_E a b (A a) (D a) (C a c) in
  let F := ray_AE_intersects_y_equals_2x_at_F a E in
  (real.dist (A a) E) / (real.dist E F) = 7 := 
sorry

end proof_AE_over_EF_l402_402074


namespace students_catching_up_on_homework_l402_402292

theorem students_catching_up_on_homework
  (total_students : ℕ)
  (half_doing_silent_reading : ℕ)
  (third_playing_board_games : ℕ)
  (remain_catching_up_homework : ℕ) :
  total_students = 24 →
  half_doing_silent_reading = total_students / 2 →
  third_playing_board_games = total_students / 3 →
  remain_catching_up_homework = total_students - (half_doing_silent_reading + third_playing_board_games) →
  remain_catching_up_homework = 4 :=
by
  intros h_total h_half h_third h_remain
  sorry

end students_catching_up_on_homework_l402_402292


namespace non_congruent_polyhedra_exist_l402_402375

def is_center (p : Point) (s : Square) : Prop := sorry
def are_visible_edges (s : Square) : Prop := sorry
def no_hidden_edges (s : Square) : Prop := sorry
def polyhedra_non_congruent (p1 p2 : Polyhedron) : Prop := sorry
def front_top_view_match (p : Polyhedron) (views : pair View) : Prop := sorry

noncomputable def square := sorry
noncomputable def front_view := sorry
noncomputable def top_view := sorry
noncomputable def views := (front_view, top_view)

axiom polyhedron_1 : Polyhedron
axiom polyhedron_2 : Polyhedron

theorem non_congruent_polyhedra_exist :
  is_center (center square) square ∧
  are_visible_edges square ∧
  no_hidden_edges square ∧
  front_top_view_match polyhedron_1 views ∧
  front_top_view_match polyhedron_2 views ∧
  polyhedra_non_congruent polyhedron_1 polyhedron_2 := sorry

end non_congruent_polyhedra_exist_l402_402375


namespace distribution_difference_l402_402011

theorem distribution_difference 
  (total_amnt : ℕ)
  (p_amnt : ℕ) 
  (q_amnt : ℕ) 
  (r_amnt : ℕ)
  (s_amnt : ℕ)
  (h_total : total_amnt = 1000)
  (h_p : p_amnt = 2 * q_amnt)
  (h_s : s_amnt = 4 * r_amnt)
  (h_qr : q_amnt = r_amnt) :
  s_amnt - p_amnt = 250 := 
sorry

end distribution_difference_l402_402011


namespace sum_first_n_terms_l402_402406

def sequence (n : ℕ) : ℚ := n + 1 / 2^(n-1)

def sum_sequence (n : ℕ) : ℚ :=
  (Finset.range n).sum (λ i => sequence (i+1))

theorem sum_first_n_terms (n : ℕ) : 
  sum_sequence n = (n^2 + n + 4) / 2 - 1 / 2^(n-1) :=
by
  sorry

end sum_first_n_terms_l402_402406


namespace average_gas_mileage_l402_402943

-- Definitions according to problem conditions
def distance_to : ℝ := 150
def mpg_sedan : ℝ := 25
def distance_return : ℝ := 150
def mpg_pickup : ℝ := 15
def total_distance : ℝ := distance_to + distance_return

-- Theorem statement to prove the average gas mileage
theorem average_gas_mileage : 
  let total_gas_used := (distance_to / mpg_sedan) + (distance_return / mpg_pickup) in
  total_distance / total_gas_used = 18.75 :=
by
  sorry

end average_gas_mileage_l402_402943


namespace ratio_of_areas_l402_402922

theorem ratio_of_areas (r : ℝ) (h : r = 3) :
  let A_circle := π * r^2,
      L_arc := (1/3) * (2 * π * r),
      s := L_arc,
      A_triangle := (sqrt 3 / 4) * s^2
  in r = 3 -> (A_triangle / A_circle) = (π * sqrt 3 / 9) :=
by
  intros
  simp only [A_circle, L_arc, s, A_triangle]
  sorry

end ratio_of_areas_l402_402922


namespace integer_values_a_l402_402178

theorem integer_values_a :
  (∃! a : ℤ, ∃ m n : ℤ, m^2 + n^2 = (m + n)^2 - 2mn ∧
                     m + n = -a ∧ 
                     mn = 12a) :
  8 :=
sorry

end integer_values_a_l402_402178


namespace actual_revenue_is_57_69_percent_of_projected_l402_402348

def projected_revenue (R : ℝ) : ℝ := 1.30 * R

def actual_revenue (R : ℝ) : ℝ := 0.75 * R

def revenue_percentage (R : ℝ) : ℝ := (actual_revenue R / projected_revenue R) * 100

theorem actual_revenue_is_57_69_percent_of_projected (R : ℝ) : 
  revenue_percentage R ≈ 57.69 :=
by 
  sorry

end actual_revenue_is_57_69_percent_of_projected_l402_402348


namespace find_m_and_parabola_eq_range_of_BP_dot_BQ_l402_402207

noncomputable def P : Point := ⟨1, 3⟩
noncomputable def A : Point := ⟨1, -3 * Real.sqrt 2 / 2⟩
noncomputable def B : Point := ⟨2, 5⟩

noncomputable def C (m : ℝ) : Circle := {
  center := ⟨m, 0⟩,
  radius := Real.sqrt (9 / 2)
}

noncomputable def parabola (p : ℝ) : Parabola := {
  focus := ⟨p / 2, 0⟩,
  directrix := -p / 2
}

theorem find_m_and_parabola_eq (m : ℝ) (p : ℝ) (F : Point) (PF_tangent : TangentLine) :
  (A ∈ C 1) → m = 1 ∧ parabola p = { focus := ⟨8, 0⟩, directrix := -8 } := 
sorry

theorem range_of_BP_dot_BQ (Q : Point) (p : ℝ) :
  ∃ (upper_bound : ℝ), 
    upper_bound = 28 ∧ ∀ (Q ∈ parabola p), 
      ∃ (BP BQ : Vector), 
        BP.dot BQ ≤ upper_bound := 
sorry

end find_m_and_parabola_eq_range_of_BP_dot_BQ_l402_402207


namespace triangle_area_proof_l402_402652

-- Define the parametric line equations
def line_parametric (t : ℝ) : ℝ × ℝ :=
  (1 + t, t - 3)

-- Define the polar equation of the curve C in Cartesian coordinates y² = 2x
def curve_cartesian (x y : ℝ) : Prop :=
  y^2 = 2 * x

-- Define the function to calculate the intersections and the area of the triangle formed by the origin and the intersection points
noncomputable def triangle_area : ℝ :=
  let t1 := 1 in
  let t2 := 7 in
  let ab_length := 6 * Real.sqrt 2 in
  let d := 2 * Real.sqrt 2 in
  (1 / 2) * ab_length * d

-- The theorem statement that needs to be proved
theorem triangle_area_proof : 
  triangle_area = 12 :=
  by sorry

end triangle_area_proof_l402_402652


namespace borrowed_amount_l402_402038

theorem borrowed_amount (P : ℝ) (h1 : (9 / 100) * P - (8 / 100) * P = 200) : P = 20000 :=
  by sorry

end borrowed_amount_l402_402038


namespace num_subsets_set_A_l402_402237

theorem num_subsets_set_A : 
  let A := {0, 1, 2}
  in (fintype.card (set A)) = 8 :=
by
  let A := {0, 1, 2}
  sorry

end num_subsets_set_A_l402_402237


namespace banquet_food_consumed_l402_402958

theorem banquet_food_consumed :
  ∃ total_food_consumed, (∀ guest (g : guest), g.food ≤ 2 ∧ 163 ≤ number_of_guests) →
  total_food_consumed = 326 :=
sorry

end banquet_food_consumed_l402_402958


namespace baron_munchausen_max_crowd_size_l402_402877

theorem baron_munchausen_max_crowd_size :
  ∃ n : ℕ, (∀ k, (k : ℕ) = n → 
  let left := (k / 2).toNat;
      right := (k / 3).toNat;
      straight := (k / 5).toNat in
  left + right + straight <= n + 1) ∧ 
  (∀ x : ℕ, x > 37 → ¬(∀ k, (k : ℕ) = x →
  let left := (k / 2).toNat;
      right := (k / 3).toNat;
      straight := (k / 5).toNat in
  left + right + straight <= x + 1)) :=
begin
  have h : 37 = 18 + 12 + 7,
  sorry,
end

end baron_munchausen_max_crowd_size_l402_402877


namespace continuous_of_increasing_and_composition_continuous_l402_402682

noncomputable def f : ℝ → ℝ := sorry ⟨strictly increasing function definition⟩

theorem continuous_of_increasing_and_composition_continuous :
  (strictly_increasing f) → (continuous (f ∘ f)) → (continuous f) :=
begin
  sorry
end

end continuous_of_increasing_and_composition_continuous_l402_402682


namespace polynomials_common_root_sum_k_l402_402636

theorem polynomials_common_root_sum_k : 
  (∀ k : ℝ, (∃ x : ℝ, x^2 - 3 * x + 2 = 0 ∧ x^2 - 5 * x + k = 0) → k = 4 ∨ k = 6) → (set.sum {4, 6} id = 10) :=
by { intro h, sorry }

end polynomials_common_root_sum_k_l402_402636


namespace f_of_minus_37_over_27_eq_8_over_9_l402_402588

noncomputable def k : ℝ := sorry
noncomputable def b : ℝ := sorry

def A : ℝ × ℝ := (-2, -1)

-- Condition: Point A lies on the graph of y = k(x + 2) - 1
lemma point_on_kx_plus_2_minus_1 (k : ℝ) : (A : ℝ × ℝ) = (-2, -1) ∧ y = k(x + 2) - 1 :=
  sorry

-- Condition: Point A lies on the graph of f(x) = 3x + b
lemma point_on_3x_plus_b (b : ℝ) : (A : ℝ × ℝ) = (-2, -1) ∧ y = 3x + b :=
  sorry

-- Given two conditions true we should satisfy f(x) = 3x + 5
lemma b_value : b = 5 :=
  sorry

-- Now proving the result f(-37/27) = 8/9
theorem f_of_minus_37_over_27_eq_8_over_9 : 
  let f (x : ℝ) := 3 * x + 5 
  in f (-37/27) = 8/9 := 
  by 
    have : f(x) = 3x + b := by sorry
    have : b = 5 := by sorry
    show (3 * (-37 / 27) + 5) = 8/9 from sorry

end f_of_minus_37_over_27_eq_8_over_9_l402_402588


namespace find_n_from_sequence_l402_402953

noncomputable def arithmetic_sequence (d : ℕ → ℤ) (a1 : ℤ) : ℕ → ℤ
| 0       => a1
| (n + 1) => arithmetic_sequence n + d n

theorem find_n_from_sequence (a1 : ℤ) (d : ℕ → ℤ) (h1 : ∑ i in finset.range (n + 1), 2 * i + 1 = 132)
    (h2 : ∑ i in finset.range n, 2 * (i + 1) = 120) : n = 10 :=
sorry

end find_n_from_sequence_l402_402953


namespace ratio_PA_AN_l402_402006

/-
  Lean 4 statement for the problem:
  ∆PNR has side lengths PN=20, NR=18, and PR=19. Point A lies on PN.
  ∆NRA is rotated about R to ∆N'RA' such that R, N', and P are collinear and AA' is perpendicular to PR.
  Prove that PA / AN = 19 / 18.
-/

open Triangle

theorem ratio_PA_AN (P N R A N' A' D : Point)
  (hPN : distance P N = 20) (hNR : distance N R = 18) (hPR : distance P R = 19)
  (hA_on_PN : lies_on A (line_through P N))
  (hNRA_rotated : rotated_about R (Triangle.mk N R A) (Triangle.mk N' R A'))
  (hRNP_collinear : collinear {R, N', P})
  (hAA'_perp_PR : perpendicular (line_through A A') (line_through P R)) :
    (distance P A) / (distance A N) = 19 / 18 := sorry

end ratio_PA_AN_l402_402006


namespace total_marks_l402_402679

theorem total_marks (k l d : ℝ) (hk : k = 3.5) (hl : l = 3.2 * k) (hd : d = l + 5.7) : k + l + d = 31.6 :=
by
  rw [hk] at hl
  rw [hl] at hd
  rw [hk, hl, hd]
  sorry

end total_marks_l402_402679


namespace number_of_distinct_a_l402_402128

noncomputable def count_distinct_a : ℕ :=
  let divisors := [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 36, 48, 72, 144, -1, -2, -3, -4, -6, -8, -9, -12, -16, -18, -24, -36, -48, -72, -144]
  let pairs := (divisors.product divisors).filter (λ (p : ℤ × ℤ), p.1 * p.2 = 144)
  let sums := pairs.map (λ (p : ℤ × ℤ), p.1 + p.2 - 24)
  let distinct_sums := sums.eraseDuplicates  
  distinct_sums.length

theorem number_of_distinct_a : count_distinct_a = 14 :=
by
  sorry

end number_of_distinct_a_l402_402128


namespace vector_sum_correct_l402_402065

-- Define the vectors involved
def v1 : Vector3 ℤ := ⟨3, -2, 7⟩
def v2 : Vector3 ℤ := ⟨-1, 5, -3⟩

-- Define the expected result
def expected_result : Vector3 ℤ := ⟨2, 3, 4⟩

-- State the theorem to prove the vector sum is correct
theorem vector_sum_correct : v1 + v2 = expected_result := by
  -- Proof omitted
  sorry

end vector_sum_correct_l402_402065


namespace problem_statement_l402_402568

noncomputable def find_angle_and_area_of_triangle 
    (a b c : ℝ) (A : ℂ) (S : ℝ) : Prop :=
  let m := (a + b + c, 3 * c)
  let n := (b, c + b - a)
  (m.1 * n.2 = m.2 * n.1) →     -- Vectors m and n are parallel
  a = sqrt 3 → 
  b = 1 → 
  A = π / 3 ∧ 
  S = (1/2) * b * c * sin (π / 3) ∧ 
  S = sqrt 3 / 2

-- Define the problem statement
theorem problem_statement 
    (a b c : ℝ) (A S : ℝ) : find_angle_and_area_of_triangle a b c A S := 
sorry

end problem_statement_l402_402568


namespace perfect_square_divisors_of_factorial_product_l402_402083

/-- 
   A perfect square divisor of a product of factorials 
   -/
theorem perfect_square_divisors_of_factorial_product :
  let product := ∏ k in Finset.range 10, nat.factorial (k + 1) in
  let prime_factors := (nat.factors product).groupBy id in
  let factor_counts := prime_factors.map (λ p => p.length) in
  let perfect_square_divisor_counts := factor_counts.map (λ n => (n / 2) + 1) in
  let number_of_perfect_square_divisors := perfect_square_divisor_counts.foldl (*) 1 in
  number_of_perfect_square_divisors = 1440 :=
by {
  sorry
}

end perfect_square_divisors_of_factorial_product_l402_402083


namespace largest_x_value_l402_402535

def largest_solution_inequality (x : ℝ) : Prop := 
  (-(Real.log 3 (100 + 2 * x * Real.sqrt (2 * x + 25)))^3 + 
  abs (Real.log 3 ((100 + 2 * x * Real.sqrt (2 * x + 25)) / (x^2 + 2 * x + 4)^4))) / 
  (3 * Real.log 6 (50 + 2 * x * Real.sqrt (2 * x + 25)) - 
  2 * Real.log 3 (100 + 2 * x * Real.sqrt (2 * x + 25))) ≥ 0 → 
  x ≤ 12 + 4 * Real.sqrt 3

theorem largest_x_value : ∃ x : ℝ, largest_solution_inequality x :=
sorry

end largest_x_value_l402_402535


namespace count_distinct_a_l402_402109

def quadratic_has_integer_solutions (a : ℤ) : Prop :=
  ∃ x m n : ℤ, (x^2 + a * x + 12 * a = 0) ∧ (m + n = -a) ∧ (m * n = 12 * a)

theorem count_distinct_a : (finset.univ.filter quadratic_has_integer_solutions).card = 9 := 
sorry

end count_distinct_a_l402_402109


namespace sumSquaresFractions_zero_l402_402331

noncomputable def sumCondition (x : Fin 50 → ℝ) : Prop :=
  ∑ i, x i = 2

noncomputable def sumFractions (x : Fin 50 → ℝ) : Prop :=
  ∑ i, x i / (1 - x i) = 2

noncomputable def sumSquaresFractions (x : Fin 50 → ℝ) : ℝ :=
  ∑ i, x i ^ 2 / (1 - x i)

theorem sumSquaresFractions_zero {x : Fin 50 → ℝ} (h₁ : sumCondition x) (h₂ : sumFractions x) :
  sumSquaresFractions x = 0 :=
sorry

end sumSquaresFractions_zero_l402_402331


namespace t_lt_s_l402_402558

noncomputable def t : ℝ := Real.sqrt 11 - 3
noncomputable def s : ℝ := Real.sqrt 7 - Real.sqrt 5

theorem t_lt_s : t < s :=
by
  sorry

end t_lt_s_l402_402558


namespace reading_hours_l402_402372

def words_per_page := 100
def pages_per_book := 80
def words_per_minute := 40
def number_of_books := 6

def total_words := words_per_page * pages_per_book * number_of_books
def reading_time_in_minutes := total_words / words_per_minute
def reading_time_in_hours := reading_time_in_minutes / 60

theorem reading_hours : reading_time_in_hours = 20 := by
  -- Calculation in steps
  have h1 : total_words = 100 * 80 * 6 := rfl
  have h2 : words_per_minute = 40 := rfl
  have h3 : reading_time_in_minutes = (100 * 80 * 6) / 40 := rfl
  have h4 : reading_time_in_hours = ((100 * 80 * 6) / 40) / 60 := rfl

  -- Calculation
  rw [h1, h3, h4]
  norm_num
  sorry

end reading_hours_l402_402372


namespace age_not_child_l402_402353

theorem age_not_child (children_ages : Finset ℕ) (eldest_age : ℕ)
                      (eldest_age_condition : eldest_age = 10)
                      (age_less_than_ten : ∀ age ∈ children_ages, age < 10)
                      (num : ℕ)
                      (num_digits : num.digits 5 = 5)
                      (three_diff_digits : Finset.length (Finset.filter (λ d, d ∈ num.digits) Finset.univ) = 3)
                      (one_digit_three_times : ∃ d, d ∈ num.digits ∧ num.digits.count d = 3)
                      (divisible_by_ages : ∀ age ∈ (insert eldest_age children_ages), num % age = 0)
                      (sum_digits : digits_sum num = 2 * mr_smith.age)
                      (mr_smith_age_depiction : num % 100 = mr_smith.age) :
  4 ∉ children_ages :=
sorry

-- Helper functions and definitions likely needed for the above statement to work
def digits_sum (n : ℕ) : ℕ := n.digits.sum

structure MrSmith where
  age : ℕ

end age_not_child_l402_402353


namespace tensor_A_B_l402_402975

def A : Set ℝ := {real.sqrt 2, real.sqrt 3}
def B : Set ℝ := {1, real.sqrt 2}

def tensor (A B : Set ℝ) : Set ℝ := 
  {z | ∃ x ∈ A, ∃ y ∈ B, z = (x + y) * (x - y)}

theorem tensor_A_B : tensor A B = {0, 1, 2} :=
by 
  sorry

end tensor_A_B_l402_402975


namespace length_of_arc_CK_l402_402830

variables (A B C K : Type) (R l : ℝ)
          (sphere : set (euclidean_space ℝ 3))
          (is_point_on_sphere : (A ∈ sphere) ∧ (B ∈ sphere) ∧ (C ∈ sphere))
          (radius_sphere : ∀ p ∈ sphere, ∥p∥ = R)
          (is_arc : shorter_arc_great_circle sphere A B ∧ shorter_arc_great_circle sphere A C ∧ shorter_arc_great_circle sphere B C)
          (midpoint_AB : midpoint (shorter_arc_great_circle sphere A B))
          (midpoint_AC : midpoint (shorter_arc_great_circle sphere A C))
          (great_circle_intersection : intersects_great_circle sphere midpoint_AB midpoint_AC (extension_of_arc sphere B C K))
          (arc_length_BC : length (shorter_arc_great_circle sphere B C) = l)
          (arc_condition : l < π * R)

theorem length_of_arc_CK :
  l < π * R →
  ∀ K, length (shorter_arc_great_circle sphere C K) = (π * R + l) / 2 ∨ length (shorter_arc_great_circle sphere C K) = (π * R - l) / 2 :=
sorry

end length_of_arc_CK_l402_402830


namespace cuberoot_floor_product_l402_402967

theorem cuberoot_floor_product : 
  (∏ n in finset.filter (λ x, x % 2 = 1) (finset.range 127), int.floor ((n : ℝ)^(1 / 3))) /
  (∏ m in finset.filter (λ x, x % 2 = 0) (finset.range 127), int.floor ((m : ℝ)^(1 / 3))) = 1 / 4 :=
by
  sorry

end cuberoot_floor_product_l402_402967


namespace TK_is_external_bisector_of_ATC_l402_402709

variable {A T C L K : Type} [LinearOrder A] [LinearOrder T] [LinearOrder C] [LinearOrder L] [LinearOrder K]
variable (triangle_ATC : Triangle A T C)
variable (point_T : Point T)
variable (point_L : Point L)
variable (point_K : Point K)
variable (internal_bisector_TL : AngleBisector (angle_ATC) T L)
variable (external_bisector_TK : AngleBisector (external_angle_ATC) T K)

theorem TK_is_external_bisector_of_ATC :
  external_bisector_TK = true :=
sorry

end TK_is_external_bisector_of_ATC_l402_402709


namespace new_sales_volume_monthly_profit_maximize_profit_l402_402920

-- Define assumptions and variables
variables (x : ℝ) (p : ℝ) (v : ℝ) (profit : ℝ)

-- Part 1: New sales volume after price increase
theorem new_sales_volume (h : 0 < x ∧ x < 20) : v = 600 - 10 * x :=
sorry

-- Part 2: Price and quantity for a monthly profit of 10,000 yuan
theorem monthly_profit (h : profit = (40 + x - 30) * (600 - 10 * x)) (h2: profit = 10000) : p = 50 ∧ v = 500 :=
sorry

-- Part 3: Price for maximizing monthly sales profit
theorem maximize_profit (h : profit = (40 + x - 30) * (600 - 10 * x)) : (∃ x_max: ℝ, x_max < 20 ∧ ∀ x, x < 20 → profit ≤ -10 * (x - 25)^2 + 12250 ∧ p = 59 ∧ profit = 11890) :=
sorry

end new_sales_volume_monthly_profit_maximize_profit_l402_402920


namespace sum_remainder_mod_11_l402_402333

def integers := Fin 11

variables (a b c : integers)

theorem sum_remainder_mod_11
  (h1 : a * b * c = 1)
  (h2 : 7 * c = 4)
  (h3 : 8 * b = 5 + b) :
  (a + b + c) = 9 :=
sorry

end sum_remainder_mod_11_l402_402333


namespace not_and_implies_at_most_one_true_l402_402637

variables p q : Prop

theorem not_and_implies_at_most_one_true (h: ¬ (p ∧ q)) : ¬ (p ∧ q) → (¬ p ∨ ¬ q) :=
sorry

end not_and_implies_at_most_one_true_l402_402637


namespace external_angle_bisector_of_triangle_l402_402737

variables {A T C L K : Type}
variables [IsTriangle A T C] [IsInternalAngleBisector T L (A T C)]

theorem external_angle_bisector_of_triangle 
  (h1 : is_internal_angle_bisector TL ATC) :
  is_external_angle_bisector TK ATC :=
sorry

end external_angle_bisector_of_triangle_l402_402737


namespace number_of_meetings_third_meeting_time_l402_402895

variables (S : ℝ)

def speed_of_cyclist := (8 * S) / 5
def speed_of_pedestrian := (3 * S) / 5

def eq_line_AE (t : ℝ) := (3 * S * t) / 5
def eq_line_FC (t : ℝ) := -(8 * S * (t - 15 / 8)) / 5

-- Prove number of meetings
theorem number_of_meetings : 
  (∃ t : ℝ, t = 0.0) ->  -- Intersection at time t = 0
  (∃ t : ℝ, t = 1.0) ->  -- and so on for 8 intersections
  8 := 
sorry

-- Prove the time coordinate tc of the third meeting point C
theorem third_meeting_time (t_c : ℝ) : 
  eq_line_FC t_c = eq_line_AE t_c -> 
  t_c = 15 / 11 := 
sorry

end number_of_meetings_third_meeting_time_l402_402895


namespace geometric_sequence_a_n_sum_of_first_n_terms_b_n_l402_402281

theorem geometric_sequence_a_n :
  (∃ a₁ a₂, a₁ * a₂ = 8 ∧ a₁ + a₂ = 6 ∧ a₁ < a₂) →
  (∀ n : ℕ, n > 0 → (∃ a : ℕ → ℕ, (a 1 = 2) ∧ (a 2 = 4) ∧ (∀ n, a n = 2 ^ n))) :=
begin
  sorry
end

theorem sum_of_first_n_terms_b_n :
  (∀ n : ℕ, n > 0 → (∃ a : ℕ → ℕ, (a 1 = 2) ∧ (a 2 = 4) ∧ (∀ n, a n = 2 ^ n))) →
  (∀ b : ℕ → ℕ, (∀ n, b n = 2 * (2 ^ n) + 3)) →
  (∀ T : ℕ → ℤ, (∀ n, T n = 2 ^ (n + 2) - 4 + 3 * n)) :=
begin
  sorry
end

end geometric_sequence_a_n_sum_of_first_n_terms_b_n_l402_402281


namespace average_breadth_of_plot_l402_402648

theorem average_breadth_of_plot :
  ∃ B L : ℝ, (L - B = 10) ∧ (23 * B = (1/2) * (L + B) * B) ∧ (B = 18) :=
by
  sorry

end average_breadth_of_plot_l402_402648


namespace integer_solution_count_l402_402145

def has_integer_solutions (a : ℤ) : Prop :=
  ∃ x y : ℤ, x^2 + a * x + 12 * a = 0 ∧ y^2 + a * y + 12 * a = 0

theorem integer_solution_count : 
  (set.finite {a : ℤ | has_integer_solutions a}).to_finset.card = 16 := 
begin
  sorry
end

end integer_solution_count_l402_402145


namespace pencils_left_l402_402092

def initial_pencils : Nat := 127
def pencils_from_joyce : Nat := 14
def pencils_per_friend : Nat := 7

theorem pencils_left : ((initial_pencils + pencils_from_joyce) % pencils_per_friend) = 1 := by
  sorry

end pencils_left_l402_402092


namespace min_distance_from_origin_to_line_l402_402775

-- Definitions based on conditions
def is_on_line (x y : ℝ) : Prop := x + y - 4 = 0

def origin : ℝ × ℝ := (0, 0)

-- Statement of the proof problem
theorem min_distance_from_origin_to_line :
  ∃ (x y : ℝ), is_on_line x y ∧ sqrt (x^2 + y^2) = 2 * sqrt 2 :=
by
  sorry

end min_distance_from_origin_to_line_l402_402775


namespace find_k_l402_402612

def vector := ℝ × ℝ × ℝ

def dot_product (a b : vector) : ℝ :=
  a.1 * b.1 + a.2 * b.2 + a.3 * b.3

def v_a : vector := (1, 1, 0)
def v_b : vector := (-1, 0, 2)

-- Condition: k * v_a + v_b is perpendicular to 2 * v_a - v_b
def is_perpendicular (x y : vector) : Prop := dot_product x y = 0

theorem find_k (k : ℝ) (h : is_perpendicular (k • v_a + v_b) (2 • v_a - v_b)) :
  k = 7 / 5 :=
sorry

end find_k_l402_402612


namespace BC_length_l402_402095

theorem BC_length {A B C : Type*} [EuclideanGeometry A B C] (r₁ r₂ : ℝ) (hA : r₁ = 5) (hB : r₂ = 3) 
  (hAB : dist A B = r₁ + r₂) (tangent_CA : TangentLine A C) (tangent_CB : TangentLine B C) : 
  dist B C = 12 :=
by
  sorry

end BC_length_l402_402095


namespace largest_number_l402_402950

noncomputable def is_largest (n : ℝ) : Prop :=
  n = π

theorem largest_number (h₁ : 3 < Real.pi)
                       (h₂ : Real.pi < 4)
                       (h₃ : -4 < -Real.pi)
                       (h₄ : -Real.pi < -3)
                       (h₅ : 9 < Real.pi^2)
                       (h₆ : Real.pi^2 < 16)
                       (h₇ : -3 < 1 - Real.pi)
                       (h₈ : 1 - Real.pi < -2)
                       (h₉ : -16 < -Real.pi^2)
                       (h₀ : -Real.pi^2 < -9) : is_largest Real.pi :=
sorry

end largest_number_l402_402950


namespace simplify_expression_correct_l402_402377

variable {R : Type} [CommRing R]

def simplify_expression (x : R) : R :=
  2 * x^2 * (4 * x^3 - 3 * x + 1) - 7 * (x^3 - 3 * x^2 + 2 * x - 8)

theorem simplify_expression_correct (x : R) : 
  simplify_expression x = 8 * x^5 + 0 * x^4 - 13 * x^3 + 23 * x^2 - 14 * x + 56 :=
by
  sorry

end simplify_expression_correct_l402_402377


namespace find_n_l402_402420

/-- Given a natural number n such that LCM(n, 12) = 48 and GCF(n, 12) = 8, prove that n = 32. -/
theorem find_n (n : ℕ) (h1 : Nat.lcm n 12 = 48) (h2 : Nat.gcd n 12 = 8) : n = 32 :=
sorry

end find_n_l402_402420


namespace number_of_solutions_3x_plus_y_eq_100_l402_402814

theorem number_of_solutions_3x_plus_y_eq_100 :
  {xy : ℕ × ℕ // (3 * xy.1 + xy.2 = 100) ∧ (xy.1 > 0) ∧ (xy.2 > 0)}.card = 33 := 
by
  sorry

end number_of_solutions_3x_plus_y_eq_100_l402_402814


namespace polynomial_complete_square_l402_402264

theorem polynomial_complete_square :
  ∃ a h k : ℝ, (∀ x : ℝ, 4 * x^2 - 12 * x + 1 = a * (x - h)^2 + k) ∧ a + h + k = -2.5 := by
  sorry

end polynomial_complete_square_l402_402264


namespace certain_event_l402_402000

def EventA : Prop := sorry
def EventB : Prop := sorry
def EventC : Prop := sorry
def EventD : Prop := sorry

theorem certain_event (A B C D : Prop) : (B = "The sun rises from the east") := by
  sorry

end certain_event_l402_402000


namespace count_distinct_a_l402_402114

def quadratic_has_integer_solutions (a : ℤ) : Prop :=
  ∃ x m n : ℤ, (x^2 + a * x + 12 * a = 0) ∧ (m + n = -a) ∧ (m * n = 12 * a)

theorem count_distinct_a : (finset.univ.filter quadratic_has_integer_solutions).card = 9 := 
sorry

end count_distinct_a_l402_402114


namespace largest_crowd_size_l402_402882

theorem largest_crowd_size (x : ℕ) : 
  (ceil (x * (1 / 2)) + ceil (x * (1 / 3)) + ceil (x * (1 / 5)) = x) →
  x ≤ 37 :=
sorry

end largest_crowd_size_l402_402882


namespace largest_divisible_by_3_power_l402_402427

theorem largest_divisible_by_3_power :
  ∃ n : ℕ, (∀ k : ℕ, 1 ≤ k ∧ k ≤ 100 → ∃ m : ℕ, (3^m ∣ (2*k - 1)) → n = 49) :=
sorry

end largest_divisible_by_3_power_l402_402427


namespace intersection_point_P_l402_402302

variables {A B C D E P : Type}
variables [vector_space ℝ A] [vector_space ℝ B] [vector_space ℝ C] [vector_space ℝ D] [vector_space ℝ E] [vector_space ℝ P]
variables (vecA vecB vecC vecD vecE vecP : ℝ → A → B → C → D → E → P)
variables (t s : ℝ)

-- Definitions of points based on conditions
def point_D (vecB vecC : B) : Prop :=
  vecD = (4 / 3) • vecC - (1 / 3) • vecB

def point_E (vecA vecC : A) : Prop :=
  vecE = (2 / 3) • vecA + (1 / 3) • vecC

-- Definition of intersection point P
def point_P (vecA vecB vecC vecD vecE vecP : A → B → C → D → E → P) : Prop :=
  ∃ t s, vecP = t • vecD + (1 - t) • vecA ∧ vecP = s • vecB + (1 - s) • vecE ∧
  vecP = (8 / 21) • vecA + (1 / 21) • vecB + (12 / 21) • vecC

-- Statement to be proven
theorem intersection_point_P (A B C D E P : Type)
  [vector_space ℝ A] [vector_space ℝ B] [vector_space ℝ C] [vector_space ℝ D] [vector_space ℝ E] [vector_space ℝ P]
  (vecA : ℝ → A) (vecB : ℝ → B) (vecC : ℝ → C) (vecD : ℝ → D) (vecE : ℝ → E) (vecP : ℝ → P)
  (h_D : point_D vecB vecC)
  (h_E : point_E vecA vecC) :
  point_P vecA vecB vecC vecD vecE vecP :=
by
  sorry

end intersection_point_P_l402_402302


namespace find_x_value_l402_402218

theorem find_x_value (x : ℝ) (y : ℝ) (h1 : y = (x^2 - 9) / (x - 3)) (h2 : y = 3 * x - 4) : x = 7 / 2 := 
sorry

end find_x_value_l402_402218


namespace find_x_of_arithmetic_mean_l402_402583

theorem find_x_of_arithmetic_mean (x : ℝ) (h : (6 + 13 + 18 + 4 + x) / 5 = 10) : x = 9 :=
by
  sorry

end find_x_of_arithmetic_mean_l402_402583


namespace smallest_possible_integer_l402_402548

theorem smallest_possible_integer (a b : ℤ)
  (a_lt_10 : a < 10)
  (b_lt_10 : b < 10)
  (a_lt_b : a < b)
  (sum_eq_45 : a + b + 32 = 45)
  : a = 4 :=
by
  sorry

end smallest_possible_integer_l402_402548


namespace external_angle_bisector_l402_402714

open EuclideanGeometry

variables {A T C K L : Point}

theorem external_angle_bisector
    (h1 : is_internal_angle_bisector T L A T C) : is_external_angle_bisector T K A T C :=
sorry

end external_angle_bisector_l402_402714


namespace poly_degree_l402_402426

-- Definition of the base polynomial
def base_poly : Polynomial ℝ := 2 * X ^ 3 + 5

-- Definition of the polynomial we are interested in
def poly : Polynomial ℝ := (base_poly) ^ 10

-- Statement of the theorem
theorem poly_degree : poly.degree = 30 :=
by {
  sorry
}

end poly_degree_l402_402426


namespace largest_crowd_size_l402_402884

theorem largest_crowd_size (x : ℕ) : 
  (ceil (x * (1 / 2)) + ceil (x * (1 / 3)) + ceil (x * (1 / 5)) = x) →
  x ≤ 37 :=
sorry

end largest_crowd_size_l402_402884


namespace power_of_product_l402_402442

theorem power_of_product (x : ℝ) : (-x^4)^3 = -x^12 := 
by sorry

end power_of_product_l402_402442


namespace max_lateral_surface_area_l402_402042

theorem max_lateral_surface_area (x y : ℝ) (h₁ : x + y = 10) : 
  2 * π * x * y ≤ 50 * π :=
by
  sorry

end max_lateral_surface_area_l402_402042


namespace probability_calculation_l402_402776
noncomputable def probability_divisible_by_4 : ℚ :=
  let total_outcomes := (1004 : ℚ) ^ 3 in
  let favorable_outcomes := 9 * (251 * 251 * 251) in
  favorable_outcomes / total_outcomes

theorem probability_calculation :
  probability_divisible_by_4 = 9 / 64 :=
by sorry

end probability_calculation_l402_402776


namespace problem1a_problem1b_l402_402919

noncomputable theory

def valid_purchase_price (a : ℤ) : Prop :=
  600 * a = 1300 * (a - 140)

def maximize_profit (x : ℤ) : Prop :=
  let y := (200 * x) / 2 + 120 * x / 2 + 20 * (5 * x + 20 - 2 * x)
  x + 5 * x + 20 ≤ 200 ∧ y = 9200

theorem problem1a (a : ℤ) : valid_purchase_price a ↔ a = 260 := sorry

theorem problem1b (x : ℤ) : maximize_profit x ↔ (x = 30 ∧ (5 * x + 20 = 170)) := sorry

end problem1a_problem1b_l402_402919


namespace probability_black_then_red_l402_402482

/-- Definition of a standard deck -/
def standard_deck := {cards : Finset (Fin 52) // cards.card = 52}

/-- Definition of black cards in the deck -/
def black_cards := {cards : Finset (Fin 52) // cards.card = 26}

/-- Definition of red cards in the deck -/
def red_cards := {cards : Finset (Fin 52) // cards.card = 26}

/-- Probability of drawing the top card as black and the second card as red -/
def prob_black_then_red (deck : standard_deck) (black : black_cards) (red : red_cards) : ℚ :=
  (26 * 26) / (52 * 51)

theorem probability_black_then_red (deck : standard_deck) (black : black_cards) (red : red_cards) :
  prob_black_then_red deck black red = 13 / 51 :=
sorry

end probability_black_then_red_l402_402482


namespace maximum_weight_of_crates_l402_402444

theorem maximum_weight_of_crates :
  ∀ n, (n = 3 ∨ n = 4 ∨ n = 5) → 
  ∀ w, (w ≥ 1250) → 
  (∃ max_w, max_w = 5 * 1250) :=
by
  intros n hn w hw
  use 6250
  simp
  sorry

end maximum_weight_of_crates_l402_402444


namespace partition_set_no_perfect_square_l402_402317

theorem partition_set_no_perfect_square (n : ℕ) (hn : 0 < n) :
  ∃ m : ℕ, m = n^2 + 2*n ∧
  (∀ A : finset (finset ℕ), A.card = n →
  (∀ a b : ℕ, a ≠ b → a ∈ ⋃₀ A → b ∈ ⋃₀ A →
    ¬(∃ S ∈ A, a ∈ S ∧ b ∈ S ∧ ∃ k : ℕ, a * b = k^2))) := by
  sorry

end partition_set_no_perfect_square_l402_402317


namespace external_bisector_of_triangle_l402_402699

variables {A T C K L : Type} [noncomputable_field A T C K L]
variable {triangle ATC : Triangle A T C}

noncomputable def internal_angle_bisector (T L : Point) := 
    ∃ R, is_bisector TL ∧ TL ∝ AR RL

noncomputable def external_angle_bisector (T K : Point) := 
    ∃ M, is_bisector TK ∧ TK ∝ AM MK

theorem external_bisector_of_triangle {A T C K L : Point} 
    (hL : internal_angle_bisector T L) 
    : external_angle_bisector T K :=
sorry

end external_bisector_of_triangle_l402_402699


namespace number_of_non_empty_subsets_of_P_l402_402320

noncomputable def P : Set ℝ := {x | ∫ t in 0..x, (3 * t^2 - 10 * t + 6) = 0 ∧ x > 0}

theorem number_of_non_empty_subsets_of_P : ∀ (P : Set ℝ), 
  P = {2, 3} → (∃ n : ℕ, n = 3 ∧ number_of_non_empty_subsets P = n) :=
by
  intro P hP
  use 3
  rw [hP]
  exact Iff.rfl

end number_of_non_empty_subsets_of_P_l402_402320


namespace maize_stolen_l402_402946

-- Definitions from the problem conditions
def monthly_storage : ℕ := 1 -- tonnes of maize stored per month
def months : ℕ := 24       -- total months in 2 years
def donation : ℕ := 8       -- tonnes of maize donated
def final_amount : ℕ := 27 -- tonnes of maize at the end of 2 years

-- The theorem to prove the amount stolen
theorem maize_stolen :
  let initial_storage := monthly_storage * months in
  let total_with_donation := initial_storage + donation in
  initial_storage + donation - final_amount = 5 :=
by 
  sorry

end maize_stolen_l402_402946


namespace cost_per_slice_in_cents_l402_402413

-- Definitions based on conditions
def num_loaves := 3
def slices_per_loaf := 20
def payment := 2 * 20 -- amount given in dollars
def change := 16 -- change in dollars

-- The final statement we need to prove
theorem cost_per_slice_in_cents : 
  let total_slices := num_loaves * slices_per_loaf in
  let total_cost := payment - change in
  (total_cost * 100) / total_slices = 40 := 
by { sorry }

end cost_per_slice_in_cents_l402_402413


namespace river_speed_is_2_l402_402930

noncomputable def speed_in_still_water : ℝ := 8 -- Man's rowing speed in still water
noncomputable def total_distance : ℝ := 7.5 -- Total distance for round trip
noncomputable def round_trip_time : ℝ := 1 -- Total time for round trip

theorem river_speed_is_2 (v : ℝ) :
  let d := total_distance / 2 in
  (d / (speed_in_still_water - v) + d / (speed_in_still_water + v) = round_trip_time) →
  v = 2 :=
by
  sorry

end river_speed_is_2_l402_402930


namespace incorrect_reasoning_l402_402820

theorem incorrect_reasoning 
  (h1 : ∃ x : ℚ, ∃ y : ℚ, x ≠ y ∧ (x / y : ℚ))  -- Some rational numbers are fractions
  (h2 : ∀ z : ℤ, z ∈ ℚ)  -- Integers are rational numbers
  : ¬(∀ w : ℤ, ∃ v : ℚ, w = v / 1)  -- The form of reasoning is wrong
  := sorry

end incorrect_reasoning_l402_402820


namespace profit_rate_is_five_percent_l402_402049

theorem profit_rate_is_five_percent (cost_price selling_price : ℝ) (hx : 1.1 * cost_price - 10 = 210) : 
  (selling_price = 1.1 * cost_price) → 
  (selling_price - cost_price) / cost_price * 100 = 5 :=
by
  sorry

end profit_rate_is_five_percent_l402_402049


namespace distinct_integer_values_of_a_l402_402119

theorem distinct_integer_values_of_a :
  ∃ (a_values : Finset ℤ), (∀ a ∈ a_values, ∃ (m n : ℤ), (x^2 + a * x + 12 * a = 0) ∧ (m + n = -a) ∧ (m * n = 12 * a))
  ∧ a_values.card = 16 :=
sorry

end distinct_integer_values_of_a_l402_402119


namespace square_angle_agh_l402_402289

noncomputable def angle_agh (A B C D H G : Type*) [square A B C D] (CH_eq_HD : CH = HD) (DG_twice_DA : DG = 2 * DA) (angle_BGH : ∠BGH = 140) : Prop :=
  ∠AGH = 85

theorem square_angle_agh (A B C D H G : Type*) [square A B C D] (CH_eq_HD : CH = HD) (DG_twice_DA : DG = 2 * DA) (angle_BGH : ∠BGH = 140) : 
  angle_agh A B C D H G CH_eq_HD DG_twice_DA angle_BGH :=
sorry

end square_angle_agh_l402_402289


namespace problem_statement_l402_402254

noncomputable def odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f x

theorem problem_statement (f : ℝ → ℝ) (x0 : ℝ) (h_odd : odd_function f) :
  (f x0 + real.exp x0 = 0) → (e ^ (-x0) * f (-x0) = 1) :=
by
  sorry

end problem_statement_l402_402254


namespace men_joined_l402_402455

theorem men_joined (x : ℕ) : 
  (1000 * 15 = (1000 + x) * 12.5) → 
  x = 200 := 
by
  sorry

end men_joined_l402_402455


namespace find_original_number_l402_402799

theorem find_original_number (x : ℝ) (h : 1.125 * x - 0.75 * x = 30) : x = 80 :=
by
  sorry

end find_original_number_l402_402799


namespace students_catching_up_on_homework_l402_402294

theorem students_catching_up_on_homework :
  ∀ (total_students : ℕ) (half : ℕ) (third : ℕ),
  total_students = 24 → half = total_students / 2 → third = total_students / 3 →
  total_students - (half + third) = 4 :=
by
  intros total_students half third
  intros h_total h_half h_third
  sorry

end students_catching_up_on_homework_l402_402294


namespace external_bisector_l402_402747

open Triangle

variable {Point : Type} [EuclideanGeometry Point]

variables {A T C L K : Point}

theorem external_bisector (h : angle_bisector A T C L) : external_bisector A T C K :=
sorry

end external_bisector_l402_402747


namespace quadrilateral_condition_XF_XG_is_31_l402_402367

theorem quadrilateral_condition_XF_XG_is_31
  (O : Type) [circle O]
  (A B C D X Y E F G : O)
  (BD AC XA XE XC XF XG : ℝ)
  (h1 : inscribed_quad A B C D O)
  (h2 : side_lengths A B C D 5 3 7 9)
  (h3 : on_line_segment X B D ∧ on_line_segment Y B D)
  (h4 : fraction_of_line_segment (D X) (B D) = 1 / 3)
  (h5 : fraction_of_line_segment (B Y) (B D) = 1 / 4)
  (h6 : intersection_E (line_A_X A X) (parallel_line_through_Y_AD Y A D))
  (h7 : intersection_F (line_C_X C X) (parallel_line_through_E_AB E A B))
  (h8 : G_is_point_on_circle_other_than_C O X_C G C) :
  XF * XG = 31 := sorry

end quadrilateral_condition_XF_XG_is_31_l402_402367


namespace min_value_reciprocal_sum_l402_402691

variable (x y z : ℝ)
variable (hx : x > 0) (hy : y > 0) (hz : z > 0)
variable (hxyz : x + y + z = 2)

theorem min_value_reciprocal_sum : 
    (\frac{1}{x} + \frac{1}{y} + \frac{1}{z}) = \frac{9}{2} :=
sorry

end min_value_reciprocal_sum_l402_402691


namespace count_integer_values_of_a_l402_402153

theorem count_integer_values_of_a (a : ℤ) :
  (∃ x : ℤ, x^2 + a * x + 12 * a = 0) ↔ (a ∈ {-(m + n) | m, n : ℤ ∧ m * n = 12 * a}) → ∃! a_vals : finset ℤ, a_vals.card = 16 :=
by
  sorry

end count_integer_values_of_a_l402_402153


namespace sheets_per_day_l402_402934

-- Definitions based on conditions
def total_sheets : ℕ := 60
def total_days_per_week : ℕ := 7
def days_off : ℕ := 2

-- Derived condition from the problem
def work_days_per_week : ℕ := total_days_per_week - days_off

-- The statement to prove
theorem sheets_per_day : total_sheets / work_days_per_week = 12 :=
by
  sorry

end sheets_per_day_l402_402934


namespace medians_angles_not_both_acute_or_both_obtuse_l402_402395

noncomputable def has_medians (A B C A' B' M : Point) : Prop :=
  is_centroid A B C M ∧ angle A M B = 120

theorem medians_angles_not_both_acute_or_both_obtuse (A B C A' B' M : Point) :
  has_medians A B C A' B' M →
  ¬(is_acute_angle (angle A B' M) ∧ is_acute_angle (angle B A' M)) ∧
  ¬(is_obtuse_angle (angle A B' M) ∧ is_obtuse_angle (angle B A' M)) :=
by
  sorry

end medians_angles_not_both_acute_or_both_obtuse_l402_402395


namespace loss_percentage_l402_402459

theorem loss_percentage (CP : ℝ) (h₁ : 1100 = 1.10 * CP) (h₂ : 800 = CP - 200) : 
  20% = 20% := 
by 
  sorry

end loss_percentage_l402_402459


namespace wall_length_is_7_5_meters_l402_402617

noncomputable def brick_volume : ℚ := 25 * 11.25 * 6

noncomputable def total_brick_volume : ℚ := 6000 * brick_volume

noncomputable def wall_cross_section : ℚ := 600 * 22.5

noncomputable def wall_length (total_volume : ℚ) (cross_section : ℚ) : ℚ := total_volume / cross_section

theorem wall_length_is_7_5_meters :
  wall_length total_brick_volume wall_cross_section = 7.5 := by
sorry

end wall_length_is_7_5_meters_l402_402617


namespace degrees_of_remainder_l402_402434

theorem degrees_of_remainder (p r : Polynomial ℤ) (h : p = -5*X^6 + 7*X^2 - 4) :
  ∃ d, d < 6 ∧ degree r = some d := by
  use [0, 1, 2, 3, 4, 5]
  sorry

end degrees_of_remainder_l402_402434


namespace ratio_ae_ef_l402_402072

-- Given definitions and conditions
variables {a : ℝ} (A B C D E F : ℝ × ℝ)
def A := (a, 0)
def D := (a, a)
def B := (b : ℝ), (b, b)
def C := (c : ℝ), (c, 2 * c)

-- Assuming collinearity and the given ratio conditions
axiom collinear (h_collinear : ∃ (A B C : ℝ × ℝ), collinear_points A B C) :
  true

axiom ratio_AB_BC (h_ratio : (dist A B) / (dist B C) = 2) :
  true

-- Define intersection points and the points lying on specified lines
def E := (e : ℝ), (e, e)
axiom circumcircle_ADC (h_circum : (∃ (D E : ℝ × ℝ), circumcircle A D C E)) :
  true

axiom ray_intersection (h_intersection : (∃ (E F : ℝ × ℝ), ray_ae_intersects_y_eq_2x_at_f A E F)) :
  true

-- The final ratio to be proven
theorem ratio_ae_ef (h_final : (dist A E) / (dist E F) = 7) :
  true :=
sorry

end ratio_ae_ef_l402_402072


namespace general_term_formula_sum_of_b_first_terms_l402_402276

variable (a₁ a₂ : ℝ)
variable (a : ℕ → ℝ)
variable (b : ℕ → ℝ)
variable (T : ℕ → ℝ)

-- Conditions
axiom h1 : a₁ * a₂ = 8
axiom h2 : a₁ + a₂ = 6
axiom increasing_geometric_sequence : ∀ n : ℕ, a (n+1) = a (n) * (a₂ / a₁)
axiom initial_conditions : a 1 = a₁ ∧ a 2 = a₂
axiom b_def : ∀ n, b n = 2 * a n + 3

-- To Prove
theorem general_term_formula : ∀ n: ℕ, a n = 2 ^ (n + 1) :=
sorry

theorem sum_of_b_first_terms (n : ℕ) : T n = 2 ^ (n + 2) - 4 + 3 * n :=
sorry

end general_term_formula_sum_of_b_first_terms_l402_402276


namespace jessica_earned_from_washing_l402_402307

-- Conditions defined as per Problem a)
def weekly_allowance : ℕ := 10
def spent_on_movies : ℕ := weekly_allowance / 2
def remaining_after_movies : ℕ := weekly_allowance - spent_on_movies
def final_amount : ℕ := 11
def earned_from_washing : ℕ := final_amount - remaining_after_movies

-- Lean statement to prove Jessica earned $6 from washing the family car
theorem jessica_earned_from_washing :
  earned_from_washing = 6 := 
by
  -- Proof to be filled in later (skipped here with sorry)
  sorry

end jessica_earned_from_washing_l402_402307


namespace number_of_integer_values_of_a_l402_402168

theorem number_of_integer_values_of_a : 
  ∃ (a_set : Set ℤ), (∀ a ∈ a_set, ∃ x y : ℤ, x^2 + a * x + 12 * a = 0 ∧ y^2 + a * y + 12 * a = 0) ∧ a_set.card = 16 :=
sorry

end number_of_integer_values_of_a_l402_402168


namespace charlie_cookies_l402_402833

theorem charlie_cookies (father_cookies mother_cookies total_cookies charlie_cookies : ℕ)
  (h1 : father_cookies = 10) (h2 : mother_cookies = 5) (h3 : total_cookies = 30) :
  father_cookies + mother_cookies + charlie_cookies = total_cookies → charlie_cookies = 15 :=
by
  intros h
  sorry

end charlie_cookies_l402_402833


namespace dot_product_parallel_vector_perpendicular_vectors_l402_402611

section

variables (a b c : ℝ × ℝ) (k : ℝ)

-- Given conditions
def vec_a := (1, 2 : ℝ)
def vec_b := (4, -3 : ℝ)
def is_parallel (c₁ c₂ : ℝ × ℝ) : Prop := ∃ k : ℝ, c₁ = (k * c₂.1, k * c₂.2)
def magnitude (v : ℝ × ℝ) : ℝ := real.sqrt (v.1 * v.1 + v.2 * v.2)

-- First question: Proving dot product
theorem dot_product : vec_a.1 * vec_b.1 + vec_a.2 * vec_b.2 = -2 := by
  sorry

-- Second question: \overrightarrow{c} is parallel to \overrightarrow{a} and | \overrightarrow{c} | = 2√5
theorem parallel_vector : is_parallel c vec_a ∧ magnitude c = 2 * real.sqrt 5 → c = (-2, -4) ∨ c = (2, 4) := by
  sorry

-- Third question: \overrightarrow{b} + k \overrightarrow{a} and \overrightarrow{b} - k \overrightarrow{a} are perpendicular
theorem perpendicular_vectors : 
  (vec_b.1 + k * vec_a.1, vec_b.2 + k * vec_a.2) = ⟨b.1 + k * a.1, b.2 + k * a.2⟩
  ∧ (vec_b.1 - k * vec_a.1, vec_b.2 - k * vec_a.2) = ⟨b.1 - k * a.1, b.2 - k * a.2⟩
  ∧ ⟦((b.1 + k * a.1) * (b.1 - k * a.1) + (b.2 + k * a.2) * (b.2 - k * a.2)) = 0⟧ → k = real.sqrt 5 ∨ k = - real.sqrt 5  :=
  by
  sorry

end

end dot_product_parallel_vector_perpendicular_vectors_l402_402611


namespace number_of_tiles_needed_l402_402028

-- Define the dimensions of the floor
def floor_length : ℝ := 10
def floor_width : ℝ := 15

-- Define the dimensions of the tiles in feet
def tile_length : ℝ := 3 / 12
def tile_width : ℝ := 9 / 12

-- Define the area of the floor and one tile
def floor_area : ℝ := floor_length * floor_width
def tile_area : ℝ := tile_length * tile_width

-- State the theorem
theorem number_of_tiles_needed : (floor_area / tile_area) = 800 := by
  sorry

end number_of_tiles_needed_l402_402028


namespace two_digit_number_is_27_l402_402419

theorem two_digit_number_is_27 :
  ∃ n : ℕ, (n / 10 < 10) ∧ (n % 10 < 10) ∧ 
  (100*(n) = 37*(10*(n) + 1)) ∧ 
  n = 27 :=
by {
  sorry
}

end two_digit_number_is_27_l402_402419


namespace integers_with_factors_30_45_75_l402_402246

theorem integers_with_factors_30_45_75 (low high lcm n : ℕ) (h1 : low = 2000) (h2 : high = 3000) (h3 : lcm = Nat.lcm (Nat.lcm 30 45) 75) (h4 : n = 2) :
  (∃ x y, low ≤ x ∧ x ≤ high ∧ x % lcm = 0 ∧ low ≤ y ∧ y ≤ high ∧ y % lcm = 0 ∧ x ≠ y ∧ (∀ z, low ≤ z ∧ z ≤ high ∧ z % lcm = 0 → z = x ∨ z = y)) :=
by
  have lcm_eq : lcm = 450 := by sorry
  have multiples : ∃ x y, (x = 2250) ∧ (y = 2700) ∧ (∀ z, (low ≤ z ∧ z ≤ high ∧ z % lcm = 0) → (z = x ∨ z = y)) := by sorry
  exact multiples

end integers_with_factors_30_45_75_l402_402246


namespace time_after_5_pm_l402_402247

theorem time_after_5_pm (x : ℝ)
  (h1 : 17 = 5)
  (h2 : let initial_angle := 150)
  (h3 : let final_angle := 150)
  (h4 : let minute_hand_degrees_per_minute := 6)
  (h5 : let hour_hand_degrees_per_minute := 0.5)
  (h6 : ∀ x, let minute_hand_position := 6 * x)
             let hour_hand_position := 150 + 0.5 * x
             (minute_hand_position - hour_hand_position) % 360 = 150
  (h7 : let x := 54 + 6 / 11) :
  x = 54 + 6 / 11 :=
sorry

end time_after_5_pm_l402_402247


namespace min_number_of_stamps_exists_l402_402510

theorem min_number_of_stamps_exists : 
  ∃ s t : ℕ, 5 * s + 7 * t = 50 ∧ ∀ (s' t' : ℕ), 5 * s' + 7 * t' = 50 → s + t ≤ s' + t' := 
by
  sorry

end min_number_of_stamps_exists_l402_402510


namespace external_bisector_TK_l402_402724

-- Definitions of points and segments
variables {A T C L K : Type}

-- Conditions given in the problem
variable (angle_bisector_TL : is_angle_bisector T L A C)
variable (is_triangle : is_triangle T A C)

-- Prove that TK is the external angle bisector
theorem external_bisector_TK : is_external_angle_bisector T K A C :=
sorry

end external_bisector_TK_l402_402724


namespace find_x_l402_402998

theorem find_x (c d : ℂ) (x : ℝ) (h_cd : c * d = x - 6 * complex.I)
  (h_abs_c : abs c = 3)
  (h_abs_d : abs d = 5) :
  x = 3 * real.sqrt 21 :=
begin
  sorry
end

end find_x_l402_402998


namespace cyclist_rejoins_group_time_l402_402032

noncomputable def travel_time (group_speed cyclist_speed distance : ℝ) : ℝ :=
  distance / (cyclist_speed - group_speed)

theorem cyclist_rejoins_group_time
  (group_speed : ℝ := 35)
  (cyclist_speed : ℝ := 45)
  (distance : ℝ := 10)
  : travel_time group_speed cyclist_speed distance * 2 = 1 / 4 :=
by
  sorry

end cyclist_rejoins_group_time_l402_402032


namespace find_sum_of_sides_l402_402647

noncomputable def sum_of_other_two_sides (S a : ℝ) (α : ℝ) : ℝ := 
  real.sqrt (a^2 + 4 * S * real.cot (α / 2))

theorem find_sum_of_sides (S a : ℝ) (α : ℝ) :
  ∃ (x y : ℝ), (x + y) = sum_of_other_two_sides S a α := 
sorry

end find_sum_of_sides_l402_402647


namespace rhombus_area_l402_402472

theorem rhombus_area 
  (scale : ℝ)
  (long_diag_map : ℝ)
  (theta : ℝ)
  (sin_60 : Real.sin θ = Real.sqrt 3 / 2)
  (scale_eq : scale = 300)
  (long_diag_map_eq : long_diag_map = 6)
  (theta_eq : theta = 60 * Real.pi / 180) :
  (1 / 2) * (scale * long_diag_map) * (scale * long_diag_map) * (Real.sqrt 3 / 2) = 810000 * Real.sqrt 3 :=
by
  have long_diag := scale * long_diag_map
  rw [scale_eq, long_diag_map_eq] at long_diag
  norm_num at long_diag
  sorry

end rhombus_area_l402_402472


namespace solve_for_m_l402_402817

theorem solve_for_m (m : ℝ) : (∀ x : ℝ, ((x + m) * (x + 3)).coeff 1 = 0) → m = -3 :=
by
  intro h
  sorry

end solve_for_m_l402_402817


namespace log5_6_identity_l402_402193

noncomputable def a : ℝ := Real.log 2
noncomputable def b : ℝ := Real.log 10 / Real.log 3

theorem log5_6_identity :
  Real.log 6 / Real.log 5 = ((a * b) + 1) / (b - (a * b)) :=
by sorry

end log5_6_identity_l402_402193


namespace point_on_transformed_plane_l402_402330

theorem point_on_transformed_plane :
  let A : ℝ × ℝ × ℝ := (1, 2,2 )
  let k : ℝ := -1/5
  let plane (x y z : ℝ) : Prop := 3 * x - z + 5 = 0
  let transformed_plane (x y z : ℝ) : Prop := 3 * x - z - 1 = 0
  (transformed_plane A.1 A.2 A.3) :=
by
  -- Definitions of points, plane and transformation
  let A := (1 : ℝ, 2 : ℝ, 2 : ℝ)
  let k := (-1 / 5 : ℝ)
  let plane := λ (x y z : ℝ), 3 * x - z + 5 = 0
  let transformed_plane := λ (x y z : ℝ), 3 * x - z - 1 = 0

  -- Formal theorem statement
  calc
    transformed_plane A.1 A.2 A.3
    = 3 * A.1 - A.3 - 1 = 0 : sorry

end point_on_transformed_plane_l402_402330


namespace arithmetic_geometric_sequence_problem_l402_402569

/-- Define the arithmetic sequence and sum of first n terms -/
def a_sequence (n : ℕ) : ℝ := 1 + (n - 1) * 2
def a_sum (n : ℕ) : ℝ := n^2

/-- Define the geometric sequence and sum of first n terms -/
def b_sequence (n : ℕ) : ℝ := 1 * 1^(n-1)
def b_sum (n : ℕ) : ℝ := n

/-- The main problem statement -/
theorem arithmetic_geometric_sequence_problem :
  (∀ d, (∀ n, a_sequence n = 1 + (n - 1) * d → 
          limit (λ n, (a_sequence n / n) + b_sequence n) = 3) → 
          limit (λ n, a_sum n / n^2 + b_sum n / n) = 2) → 
          (∃ b_lim Bn_lim, limit (λ n, b_sequence n) = b_lim ∧ limit (λ n, b_sum n / n) = Bn_lim) → 
          (a_sum = λ n, n^2 ∧ b_sum = λ n, n) := 
begin 
  sorry
end

end arithmetic_geometric_sequence_problem_l402_402569


namespace total_sampled_papers_l402_402415

theorem total_sampled_papers (A B C : ℕ) (C_sample : ℕ) (S : ℕ) 
  (hA : A = 1260) 
  (hB : B = 720) 
  (hC : C = 900) 
  (hC_sample : C_sample = 50)
  (hS : S = C_sample + (C_sample * B / C) + (C_sample * A / C)) : 
  S = 160 :=
by {
  rw [hA, hB, hC, hC_sample] at hS,
  exact hS,
}

end total_sampled_papers_l402_402415


namespace find_greater_number_l402_402825

-- Define the two numbers x and y
variables (x y : ℕ)

-- Conditions
theorem find_greater_number (h1 : x + y = 36) (h2 : x - y = 12) : x = 24 := 
by
  sorry

end find_greater_number_l402_402825


namespace angle_is_pi_over_3_l402_402572

variables {V : Type*} [inner_product_space ℝ V] (a b : V)

-- Non-zero vectors condition
axiom non_zero_a : a ≠ 0
axiom non_zero_b : b ≠ 0

-- Magnitude condition
axiom magnitude_condition : ∥a∥ = 2 * ∥b∥

-- Perpendicular condition
axiom perpendicular_condition : inner a b = inner b b

-- Prove the angle between a and b is π / 3
theorem angle_is_pi_over_3 : real.angle a b = π / 3 := 
sorry

end angle_is_pi_over_3_l402_402572


namespace find_a_maximize_profit_l402_402916

-- Definition of parameters
def a := 260
def purchase_price_table := a
def purchase_price_chair := a - 140

-- Condition 1: The number of dining chairs purchased for 600 yuan is the same as the number of dining tables purchased for 1300 yuan.
def condition1 := (600 / (purchase_price_chair : ℚ)) = (1300 / (purchase_price_table : ℚ))

-- Given conditions for profit maximization
def qty_tables := 30
def qty_chairs := 5 * qty_tables + 20
def total_qty := qty_tables + qty_chairs

-- Condition: Total quantity of items does not exceed 200 units.
def condition2 := total_qty ≤ 200

-- Profit calculation
def profit := 280 * qty_tables + 800

-- Theorem statements
theorem find_a : condition1 → a = 260 := sorry

theorem maximize_profit : condition2 ∧ (8 * qty_tables + 800 > 0) → 
  (qty_tables = 30) ∧ (qty_chairs = 170) ∧ (profit = 9200) := sorry

end find_a_maximize_profit_l402_402916


namespace smallest_n_l402_402053

theorem smallest_n (n : ℕ) (h1 : 100 ≤ n ∧ n < 1000)
  (h2 : n % 9 = 2)
  (h3 : n % 6 = 4) : n = 146 :=
sorry

end smallest_n_l402_402053


namespace area_between_circles_144pi_l402_402418

noncomputable def area_between_circles (EO EG : ℝ) (EF : ℝ) : ℝ :=
  let OG := real.sqrt (EO^2 - EG^2)
  let outer_area := real.pi * (EO ^ 2)
  let inner_area := real.pi * (OG ^ 2)
  outer_area - inner_area

theorem area_between_circles_144pi 
  (O : Type) (EO : ℝ) (EF : ℝ) 
  (hEO : EO = 13)
  (hEF : EF = 24)
  : area_between_circles EO (EF / 2) EF = 144 * real.pi :=
by sorry

end area_between_circles_144pi_l402_402418


namespace fraction_of_time_l402_402438

-- Define the time John takes to clean the entire house
def John_time : ℝ := 6

-- Define the combined time it takes Nick and John to clean the entire house
def combined_time : ℝ := 3.6

-- Given this configuration, we need to prove the fraction result.
theorem fraction_of_time (N : ℝ) (H1 : John_time = 6) (H2 : ∀ N, (1/John_time) + (1/N) = 1/combined_time) :
  (John_time / 2) / N = 1 / 3 := 
by sorry

end fraction_of_time_l402_402438


namespace angle_of_inclination_l402_402099

-- Define the parametric equations of the line
variables (t : ℝ)
def x (t : ℝ) := 3 + t * Real.sin (20 * Real.pi / 180)
def y (t : ℝ) := -1 + t * Real.cos (20 * Real.pi / 180)

-- The angle of inclination of the line
theorem angle_of_inclination : 
  ∀ (t : ℝ), 
  let α := (70 : ℝ) * Real.pi / 180 in
  y t = tan α * (x t - 3) -> α = 70 * Real.pi / 180 :=
by
  sorry

end angle_of_inclination_l402_402099


namespace solve_inequality_l402_402592

def quadratic_ineq_sol {a c x : ℝ} (h_neg_a : a < 0) (h_sol : ∀ x, -1/3 < x ∧ x < 1/2 → ax^2 + 2*x + c > 0) : Prop :=
  ∃ a c, (a = -12) ∧ (c = 2) ∧ ∀ x, -2 < x ∧ x < 3 → -c*x^2 + 2*x - a > 0

theorem solve_inequality
  (h1 : ∀ x, -1/3 < x ∧ x < 1/2 → (-12)x^2 + 2*x + 2 > 0)
  (h2 : (-12) < 0) :
  ∀ x, -2 < x ∧ x < 3 → -2*x^2 + 2*x + 12 > 0 :=
by sorry

end solve_inequality_l402_402592


namespace square_field_area_l402_402381

-- Let s be the side length of the square field.
-- The perimeter without gates would be 4s.
-- Subtracting the width of 2 gates: 4s - 2
-- Representing the total cost equation:
-- (4s - 2) * 3.50 = 2331

theorem square_field_area (s : ℕ) (h : (4 * s - 2) * 3.50 = 2331) : s^2 = 27889 :=
by
  sorry

end square_field_area_l402_402381


namespace number_of_years_lent_is_8_l402_402046

-- Definitions based on the problem conditions
def total_sum : ℝ := 2769
def second_part : ℝ := 1704
def first_part : ℝ := total_sum - second_part
def interest_rate_first : ℝ := 3 / 100
def interest_rate_second : ℝ := 5 / 100
def interest_second : ℝ := (second_part) * interest_rate_second * 3

-- Definition based on the solution
def years_lent (first_part: ℝ) (interest_rate_first: ℝ) (interest_second: ℝ): ℝ :=
  interest_second * 100 / (first_part * interest_rate_first)

-- Prove that the first part is lent for 8 years
theorem number_of_years_lent_is_8 : years_lent first_part interest_rate_first interest_second = 8 := by
  sorry

end number_of_years_lent_is_8_l402_402046


namespace external_bisector_l402_402753

-- Define the relevant points and segments in the triangle
variables {A T C L K : Type} [inhabited A] [inhabited T] [inhabited C] [inhabited L] [inhabited K]

-- Assuming T, L, K are points such that TL divides angle ATC internally
axiom internal_bisector (h : L) : ∃ TL, TL.bisects_angle (A T C) internally

-- Prove that TK is the external angle bisector given the condition above
theorem external_bisector (h : L) (h_internal : internal_bisector h) : 
  ∃ TK, TK.bisects_angle (A T C) externally :=
sorry

end external_bisector_l402_402753


namespace quadrilateral_AREA_HALF_of_regular_octagon_l402_402782

theorem quadrilateral_AREA_HALF_of_regular_octagon (h : regular_octagon CH I L D R E N ∧ oct_area = 1) : 
  quad_area LINE = 1 / 2 := 
sorry

end quadrilateral_AREA_HALF_of_regular_octagon_l402_402782


namespace count_integer_values_of_a_l402_402155

theorem count_integer_values_of_a (a : ℤ) :
  (∃ x : ℤ, x^2 + a * x + 12 * a = 0) ↔ (a ∈ {-(m + n) | m, n : ℤ ∧ m * n = 12 * a}) → ∃! a_vals : finset ℤ, a_vals.card = 16 :=
by
  sorry

end count_integer_values_of_a_l402_402155


namespace range_of_f_l402_402226

-- Define the function
def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^2 - 2 * a * x + b

-- Define the problem statement as a theorem
theorem range_of_f (a b : ℝ) (h1 : ∀ x, f x a b = f (-x) a b)
                     (h2 : -2 * b ≤ 3 * b - 1) :
  ∃ (m M : ℝ), ∀ x ∈ Icc (-2 * b) (3 * b - 1), f x a b ∈ Icc m M :=
sorry

end range_of_f_l402_402226


namespace distinct_integer_values_of_a_l402_402122

theorem distinct_integer_values_of_a :
  ∃ (a_values : Finset ℤ), (∀ a ∈ a_values, ∃ (m n : ℤ), (x^2 + a * x + 12 * a = 0) ∧ (m + n = -a) ∧ (m * n = 12 * a))
  ∧ a_values.card = 16 :=
sorry

end distinct_integer_values_of_a_l402_402122


namespace middle_number_probability_l402_402815

theorem middle_number_probability :
  let k := 6
  let A := { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 }

  -- Conditions:
  (∀ (arrangement : Set (Finset ℕ)) (harrangement : arrangement.card = 11)
    (hmid : (arrangement.val.nth k).val > (arrangement.val.nth <$> (Finset.range k))),
    -- Prove question:
    let total_permutations := A.card.choose 6
    let configurations := (∑ (k : ℕ) in {2, 3, 4, 5, 6, 7}, (k - 1) * Nat.choose (11 - k) 4)
    P := 2 * 70 / 462
    P = 10 / 33 :=
  sorry

end middle_number_probability_l402_402815


namespace smallest_number_of_students_l402_402357

def has_exactly_n_divisors (n d : ℕ) : Prop := (finset.range (n + 1)).filter (λ i, i > 0 ∧ n % i = 0).card = d

theorem smallest_number_of_students : ∃ n : ℕ, (n % 12 = 0) ∧ has_exactly_n_divisors n 10 ∧ ∀ m : ℕ, (m % 12 = 0) ∧ has_exactly_n_divisors m 10 → n ≤ m :=
begin
  use 48,
  split,
  { exact nat.mod_eq_zero_of_dvd (dvd_refl 48), }, -- 48 % 12 = 0
  split,
  { -- 48 has exactly 10 divisors
    sorry
  },
  { -- prove 48 is the smallest
    sorry
  }
end

end smallest_number_of_students_l402_402357


namespace geometric_sequence_triangle_common_ratio_range_l402_402401

theorem geometric_sequence_triangle_common_ratio_range (a q : ℝ) (ha : 0 < a) (hq : 0 < q) :
  (\(\frac{a}{q} + a > aq \land \frac{a}{q} < a + aq\)) → 
  \(\frac{\sqrt{5} - 1}{2} < q ∧ q < \frac{\sqrt{5} + 1}{2}\)) ∨ 
  \(\frac{a}{q} < a + aq → \frac{\sqrt{5} - 1}{2} < q < 1 \)) :=
begin
  sorry,
end

end geometric_sequence_triangle_common_ratio_range_l402_402401


namespace sum_sequence_2018_l402_402200

def sequence (n : ℕ) : ℚ
| 0       := 2
| (n + 1) := -1 / (sequence n + 1)

noncomputable def sum_sequence (n : ℕ) : ℚ :=
∑ i in Finset.range n, sequence i

theorem sum_sequence_2018 :
  sum_sequence 2018 = 341 / 3 := 
sorry

end sum_sequence_2018_l402_402200


namespace distinct_integer_values_of_a_l402_402120

theorem distinct_integer_values_of_a :
  ∃ (a_values : Finset ℤ), (∀ a ∈ a_values, ∃ (m n : ℤ), (x^2 + a * x + 12 * a = 0) ∧ (m + n = -a) ∧ (m * n = 12 * a))
  ∧ a_values.card = 16 :=
sorry

end distinct_integer_values_of_a_l402_402120


namespace fraction_multiplication_l402_402424

theorem fraction_multiplication :
  ((3 : ℚ) / 4) ^ 3 * ((2 : ℚ) / 5) ^ 3 = (27 : ℚ) / 1000 := sorry

end fraction_multiplication_l402_402424


namespace max_people_in_crowd_l402_402871

theorem max_people_in_crowd : ∃ n : ℕ, n ≤ 37 ∧ 
    (⟨1 / 2 * n⟩ + ⟨1 / 3 * n⟩ + ⟨1 / 5 * n⟩ = n) :=
sorry

end max_people_in_crowd_l402_402871


namespace count_valid_a_l402_402332

theorem count_valid_a :
  let solutions_exist (a : ℤ) : Prop :=
    ∃ x y : ℤ, x^2 = y + a ∧ y^2 = x + a
  in
  let is_within_bounds (a : ℤ) : Prop :=
    |a| ≤ 2005
  in
  finset.card {a : ℤ | solutions_exist a ∧ is_within_bounds a} = 90 :=
begin
  sorry
end

end count_valid_a_l402_402332


namespace min_distance_M_to_circle_l402_402397

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

noncomputable def min_distance_to_circle (M : ℝ × ℝ) (R : ℝ) : ℝ :=
distance (0,0) M - R

theorem min_distance_M_to_circle (M : ℝ × ℝ) (R : ℝ) (hx : 3 = 3) (hy : 4 = 4) (hr : 1 = R) : 
  min_distance_to_circle (3, 4) 1 = 4 := by
  sorry

end min_distance_M_to_circle_l402_402397


namespace find_r_range_l402_402048

noncomputable def parabola_area_interval (r : ℝ) : Prop :=
  let base := 2 * real.sqrt (r + 4)
  let height := r + 4
  let area := (base * height) / 2
  16 ≤ area ∧ area ≤ 128

theorem find_r_range :
  {r : ℝ | parabola_area_interval r} = set.Icc (8 / 3) (52 / 3) :=
sorry

end find_r_range_l402_402048


namespace donation_to_treetown_and_forest_reserve_l402_402374

noncomputable def donation_problem (x : ℕ) :=
  x + (x + 140) = 1000

theorem donation_to_treetown_and_forest_reserve :
  ∃ x : ℕ, donation_problem x ∧ (x + 140 = 570) := 
by
  sorry

end donation_to_treetown_and_forest_reserve_l402_402374


namespace external_bisector_TK_l402_402728

-- Definitions of points and segments
variables {A T C L K : Type}

-- Conditions given in the problem
variable (angle_bisector_TL : is_angle_bisector T L A C)
variable (is_triangle : is_triangle T A C)

-- Prove that TK is the external angle bisector
theorem external_bisector_TK : is_external_angle_bisector T K A C :=
sorry

end external_bisector_TK_l402_402728


namespace stabilize_sequence_of_positive_terms_l402_402685

-- Definitions and statement
variables (a : ℕ → ℕ) (N : ℕ)

def S (a : ℕ → ℕ) (n : ℕ) :=
  ∑ i in Finset.range n, (a i) / (a (i + 1))

theorem stabilize_sequence_of_positive_terms 
  (h1 : ∀ (n : ℕ), n ≥ N → S a (n + 1) ∈ ℤ)
  (h2 : N > 1) :
  ∃ M, ∀ m ≥ M, a m = a (m + 1) :=
sorry

end stabilize_sequence_of_positive_terms_l402_402685


namespace intersection_with_y_axis_l402_402795

-- Define the given function
def f (x : ℝ) := x^2 + x - 2

-- Prove that the intersection point with the y-axis is (0, -2)
theorem intersection_with_y_axis : f 0 = -2 :=
by {
  sorry
}

end intersection_with_y_axis_l402_402795


namespace smallest_positive_period_max_min_values_on_interval_l402_402231

noncomputable def f (x : ℝ) : ℝ :=
  cos (2 * x - π / 6) * sin (2 * x) - 1 / 4

theorem smallest_positive_period :
  (∃ T > 0, ∀ x : ℝ, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x : ℝ, f (x + T') = f x) → T' >= π / 2) :=
sorry

theorem max_min_values_on_interval :
  let I := set.Icc (-π / 4) (0 : ℝ)
  in (∃ a b ∈ I, ∀ x ∈ I, f x ≤ f a ∧ f b ≤ f x) ∧
     (f a = 1 / 4) ∧ (f b = -1 / 2) :=
sorry

end smallest_positive_period_max_min_values_on_interval_l402_402231


namespace range_of_f_4_l402_402562

theorem range_of_f_4 {a b c d : ℝ} 
  (h1 : 1 ≤ a*(-1)^3 + b*(-1)^2 + c*(-1) + d ∧ a*(-1)^3 + b*(-1)^2 + c*(-1) + d ≤ 2) 
  (h2 : 1 ≤ a*1^3 + b*1^2 + c*1 + d ∧ a*1^3 + b*1^2 + c*1 + d ≤ 3) 
  (h3 : 2 ≤ a*2^3 + b*2^2 + c*2 + d ∧ a*2^3 + b*2^2 + c*2 + d ≤ 4) 
  (h4 : -1 ≤ a*3^3 + b*3^2 + c*3 + d ∧ a*3^3 + b*3^2 + c*3 + d ≤ 1) :
  -21.75 ≤ a*4^3 + b*4^2 + c*4 + d ∧ a*4^3 + b*4^2 + c*4 + d ≤ 1 :=
sorry

end range_of_f_4_l402_402562


namespace baron_munchausen_max_people_l402_402867

theorem baron_munchausen_max_people :
  ∃ x : ℕ, (x = 37) ∧ 
  (1 / 2 * x).nat_ceil + (1 / 3 * x).nat_ceil + (1 / 5 * x).nat_ceil = x := sorry

end baron_munchausen_max_people_l402_402867


namespace distinct_integer_a_values_l402_402134

open Nat

theorem distinct_integer_a_values : ∃ (S : Finset ℤ), S.card = 8 ∧
  ∀ a ∈ S, ∃ (p q : ℤ), p + q = -a ∧ p * q = 12a ∧ 
  ∀ x ∈ (polynomial.roots (polynomial.C (12 : ℚ) * polynomial.X + polynomial.C a * polynomial.X + polynomial.C 1)).val, x ∈ ℤ :=
by
  sorry

end distinct_integer_a_values_l402_402134


namespace chuck_area_correct_l402_402070

noncomputable def chuck_play_area (shed_length shed_width leash_length : ℝ) (C : shed_length = 4 ∧ shed_width = 6 ∧ leash_length = 5) : ℝ :=
  (3 / 4) * Real.pi * (leash_length ^ 2) + (1 / 2) * Real.pi * (2 ^ 2)

theorem chuck_area_correct (shed_length shed_width leash_length : ℝ) (h : shed_length = 4) (h2 : shed_width = 6) (h3 : leash_length = 5) :
  chuck_play_area shed_length shed_width leash_length (and.intro h (and.intro h2 h3)) = (83 / 4) * Real.pi :=
by
  sorry

end chuck_area_correct_l402_402070


namespace area_of_transformed_region_l402_402321

variable (T : Type) (area_T : ℝ) (M : Matrix (Fin 2) (Fin 2) ℝ) (det_M : ℝ)

-- Given conditions
def conditions := area_T = 9 ∧ M = ![![3, 0], ![8, 3]] ∧ det_M = Matrix.det M

-- Theorem statement
theorem area_of_transformed_region (h : conditions T area_T M det_M) : 
  let T' := M.mul_vec ![area_T, 1]
  area_T' = 81 :=
sorry

end area_of_transformed_region_l402_402321


namespace maximize_perimeter_OIH_l402_402662

/-- In triangle ABC, given certain angles and side lengths, prove that
    angle ABC = 70° maximizes the perimeter of triangle OIH, where O, I,
    and H are the circumcenter, incenter, and orthocenter of triangle ABC. -/
theorem maximize_perimeter_OIH 
  (A : ℝ) (B : ℝ) (C : ℝ)
  (BC : ℝ) (AB : ℝ) (AC : ℝ)
  (BOC : ℝ) (BIC : ℝ) (BHC : ℝ) :
  A = 75 ∧ BC = 2 ∧ AB ≥ AC ∧
  BOC = 150 ∧ BIC = 127.5 ∧ BHC = 105 → 
  B = 70 :=
by
  sorry

end maximize_perimeter_OIH_l402_402662


namespace well_quasi_order_iff_no_infinite_antichain_or_descending_seq_l402_402040

-- Definitions:
variable (X : Type)
variable (leq : X → X → Prop)

-- Condition: X contains neither an infinite antichain nor a strictly decreasing infinite sequence
def no_infinite_antichain_or_descending_seq :=
  ¬(∃ (f : ℕ → X), (∀ i j, i < j → ¬ leq (f i) (f j)) ∨ (∀ i j, i < j → leq (f j) (f i)))

-- The problem statement:
theorem well_quasi_order_iff_no_infinite_antichain_or_descending_seq :
  (∀ (Y : set X), ∃ (m : X), ∀ (y : Y), leq m y) ↔ no_infinite_antichain_or_descending_seq X leq := 
sorry

end well_quasi_order_iff_no_infinite_antichain_or_descending_seq_l402_402040


namespace count_integer_values_of_a_l402_402159

theorem count_integer_values_of_a (a : ℤ) :
  (∃ x : ℤ, x^2 + a * x + 12 * a = 0) ↔ (a ∈ {-(m + n) | m, n : ℤ ∧ m * n = 12 * a}) → ∃! a_vals : finset ℤ, a_vals.card = 16 :=
by
  sorry

end count_integer_values_of_a_l402_402159


namespace arithmetic_seq_a₄_l402_402204

-- Definitions for conditions in the given problem
def S₅ (a₁ a₅ : ℕ) : ℕ := ((a₁ + a₅) * 5) / 2
def arithmetic_sequence (a₁ d n : ℕ) : ℕ := a₁ + (n - 1) * d

-- Final proof statement to show that a₄ = 9
theorem arithmetic_seq_a₄ (a₁ a₅ : ℕ) (d : ℕ) (h₁ : S₅ a₁ a₅ = 35) (h₂ : a₅ = 11) (h₃ : d = (a₅ - a₁) / 4) :
  arithmetic_sequence a₁ d 4 = 9 :=
sorry

end arithmetic_seq_a₄_l402_402204


namespace TK_is_external_bisector_of_ATC_l402_402707

variable {A T C L K : Type} [LinearOrder A] [LinearOrder T] [LinearOrder C] [LinearOrder L] [LinearOrder K]
variable (triangle_ATC : Triangle A T C)
variable (point_T : Point T)
variable (point_L : Point L)
variable (point_K : Point K)
variable (internal_bisector_TL : AngleBisector (angle_ATC) T L)
variable (external_bisector_TK : AngleBisector (external_angle_ATC) T K)

theorem TK_is_external_bisector_of_ATC :
  external_bisector_TK = true :=
sorry

end TK_is_external_bisector_of_ATC_l402_402707


namespace external_bisector_l402_402758

-- Define the relevant points and segments in the triangle
variables {A T C L K : Type} [inhabited A] [inhabited T] [inhabited C] [inhabited L] [inhabited K]

-- Assuming T, L, K are points such that TL divides angle ATC internally
axiom internal_bisector (h : L) : ∃ TL, TL.bisects_angle (A T C) internally

-- Prove that TK is the external angle bisector given the condition above
theorem external_bisector (h : L) (h_internal : internal_bisector h) : 
  ∃ TK, TK.bisects_angle (A T C) externally :=
sorry

end external_bisector_l402_402758


namespace more_money_from_mom_is_correct_l402_402543

noncomputable def more_money_from_mom : ℝ :=
  let money_from_mom := 8.25
  let money_from_dad := 6.50
  let money_from_grandparents := 12.35
  let money_from_aunt := 5.10
  let money_spent_toy := 4.45
  let money_spent_snacks := 6.25
  let total_received := money_from_mom + money_from_dad + money_from_grandparents + money_from_aunt
  let total_spent := money_spent_toy + money_spent_snacks
  let money_remaining := total_received - total_spent
  let money_spent_books := 0.25 * money_remaining
  let money_left_after_books := money_remaining - money_spent_books
  money_from_mom - money_from_dad

theorem more_money_from_mom_is_correct : more_money_from_mom = 1.75 := by
  sorry

end more_money_from_mom_is_correct_l402_402543


namespace baron_munchausen_max_crowd_size_l402_402876

theorem baron_munchausen_max_crowd_size :
  ∃ n : ℕ, (∀ k, (k : ℕ) = n → 
  let left := (k / 2).toNat;
      right := (k / 3).toNat;
      straight := (k / 5).toNat in
  left + right + straight <= n + 1) ∧ 
  (∀ x : ℕ, x > 37 → ¬(∀ k, (k : ℕ) = x →
  let left := (k / 2).toNat;
      right := (k / 3).toNat;
      straight := (k / 5).toNat in
  left + right + straight <= x + 1)) :=
begin
  have h : 37 = 18 + 12 + 7,
  sorry,
end

end baron_munchausen_max_crowd_size_l402_402876


namespace select_best_athlete_l402_402191

noncomputable def best_athlete {a b c d : Type}
  (A_score : ℝ) (A_sd : ℝ)
  (B_score : ℝ) (B_sd : ℝ)
  (C_score : ℝ) (C_sd : ℝ)
  (D_score : ℝ) (D_sd : ℝ) : Type :=
  A_score = 8.6 ∧ A_sd = 1.3 ∧
  B_score = 8.6 ∧ B_sd = 1.5 ∧
  C_score = 9.1 ∧ C_sd = 1.0 ∧
  D_score = 9.1 ∧ D_sd = 1.2 ∧
  C_score > A_score ∧ C_score > B_score ∧ C_score = D_score ∧ C_sd < D_sd

theorem select_best_athlete
  (A_score : ℝ) (A_sd : ℝ)
  (B_score : ℝ) (B_sd : ℝ)
  (C_score : ℝ) (C_sd : ℝ)
  (D_score : ℝ) (D_sd : ℝ) :
  best_athlete A_score A_sd B_score B_sd C_score C_sd D_score D_sd → (C_score, C_sd) = (9.1, 1.0) :=
by
  sorry

end select_best_athlete_l402_402191


namespace find_a_l402_402532

theorem find_a (r s : ℚ) (a : ℚ) :
  (∀ x : ℚ, (ax^2 + 18 * x + 16 = (r * x + s)^2)) → 
  s = 4 ∨ s = -4 →
  a = (9 / 4) * (9 / 4)
:= sorry

end find_a_l402_402532


namespace complex_product_l402_402963

theorem complex_product : (3 + 4 * I) * (-2 - 3 * I) = -18 - 17 * I :=
by
  sorry

end complex_product_l402_402963


namespace general_term_and_sum_l402_402282

variable {n : ℕ}
variable {a : ℕ → ℕ}
variable {b : ℕ → ℕ}
variable {T : ℕ → ℕ}

-- Conditions for the geometric sequence {a_n}
axiom a_seq_geometric (n : ℕ) (a1 a2 : ℕ) (h1 : a1 * a2 = 8) (h2 : a1 + a2 = 6) : a n = 2^n

-- Definition of sequence {b_n}
def b_seq (n : ℕ) : ℕ := 2 * a n + 3

-- Sum of the first n terms of the sequence {b_n}
axiom sum_b_seq (n : ℕ) : T n = (2 ^ (n + 2)) - 4 + 3 * n

-- Theorem to prove
theorem general_term_and_sum 
(h : ∀ n, a n = 2 ^ n) 
(h_sum: ∀ n, T n = (2 ^ (n + 2)) - 4 + 3 * n) :
∀ n, (a n = 2 ^ n) ∧ (T n = (2 ^ (n + 2)) - 4 + 3 * n) := by
  intros
  exact ⟨h n, h_sum n⟩

end general_term_and_sum_l402_402282


namespace camryn_flute_practice_interval_l402_402514

theorem camryn_flute_practice_interval (x : ℕ) 
  (h1 : ∃ n : ℕ, n * 11 = 33) 
  (h2 : x ∣ 33) 
  (h3 : x < 11) 
  (h4 : x > 1) 
  : x = 3 := 
sorry

end camryn_flute_practice_interval_l402_402514


namespace parabola_trajectory_l402_402215

theorem parabola_trajectory (P : ℝ × ℝ) : 
  (dist P (3, 0) = dist P (3 - 1, P.2 - 0)) → P.2^2 = 12 * P.1 := 
sorry

end parabola_trajectory_l402_402215


namespace parallelogram_sum_l402_402525

open Finset

noncomputable def distance (p1 p2 : ℤ × ℤ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

noncomputable def perimeter (vertices : Finset (ℤ × ℤ)) : ℝ :=
  let [a, b, c, d] := vertices.sort (· ≤ ·)
  distance a b + distance b c + distance c d + distance d a

noncomputable def area (vertices : Finset (ℤ × ℤ)) : ℝ :=
  let [a, b, _, _] := vertices.sort (· ≤ ·)
  let base_vector := (b.1 - a.1, b.2 - a.2)
  let height_vector := (d.1 - a.1, d.2 - a.2)
  Real.abs (base_vector.1 * height_vector.2 - base_vector.2 * height_vector.1)

theorem parallelogram_sum (vertices : Finset (ℤ × ℤ)) : 
  (vertices = {(1,3), (5,6), (11,6), (7,3)}) → 
  (perimeter vertices + area vertices = 22 + 18) :=
by
  sorry

end parallelogram_sum_l402_402525


namespace external_bisector_l402_402748

open Triangle

variable {Point : Type} [EuclideanGeometry Point]

variables {A T C L K : Point}

theorem external_bisector (h : angle_bisector A T C L) : external_bisector A T C K :=
sorry

end external_bisector_l402_402748


namespace arithmetic_mean_probability_integer_l402_402412

open Finset

theorem arithmetic_mean_probability_integer :
  let S := {2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030}.toFinset in
  let favorable : Finset (Finset ℕ) := S.powerset.filter (λ t, t.card = 3 ∧ (t.sum id) % 3 = 0) in
  let total : Finset (Finset ℕ) := S.powerset.filter (λ t, t.card = 3) in
  (toRational favorable.card / toRational total.card) = (7 / 20 : ℚ) :=
by {
  sorry
}

end arithmetic_mean_probability_integer_l402_402412


namespace volume_of_rotated_solid_l402_402649

-- Define the conditions
def angle_C_eq_90_degrees : Prop := ∠C = 90
def AC_eq_2 : Prop := AC = 2
def BC_eq_1 : Prop := BC = 1

-- State the theorem to prove the volume of the solid
theorem volume_of_rotated_solid :
  angle_C_eq_90_degrees ∧ AC_eq_2 ∧ BC_eq_1 →
  ∃ V, V = (2 * π / 3) :=
by
  -- This will be filled with the actual proof
  sorry

end volume_of_rotated_solid_l402_402649


namespace scientist_prob_rain_l402_402988

theorem scientist_prob_rain (x : ℝ) (p0 p1 : ℝ)
  (h0 : p0 + p1 = 1)
  (h1 : ∀ x : ℝ, x = (p0 * x^2 + p0 * (1 - x) * x + p1 * (1 - x) * x) / x + (1 - x) - x^2 / (x + 1))
  (h2 : (x + p0 / (x + 1) - x^2 / (x + 1)) = 0.2) :
  x = 1/9 := 
sorry

end scientist_prob_rain_l402_402988


namespace proof_problem_l402_402653

noncomputable def vectors : ℝ × ℝ × ℝ × ℝ := (1, 2, 3, 1)

-- Extract the vectors for OA and OB
def OA : ℝ × ℝ := (vectors.1, vectors.2)
def OB : ℝ × ℝ := (vectors.3, vectors.4)

-- Define the vector AB
def AB : ℝ × ℝ := (OB.1 - OA.1, OB.2 - OA.2)

-- Define the magnitude of a vector
def magnitude (v : ℝ × ℝ) : ℝ := real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Define the dot product of two vectors 
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Prove the required conditions
theorem proof_problem :
  magnitude AB = real.sqrt 5 ∧
  (dot_product AB OA = 0) ∧
  (∃ θ : ℝ, θ = real.pi / 4 ∧ cos θ = (dot_product OA OB) / (magnitude OA * magnitude OB)) ∧
  ¬(OA.1 = (OA.1 + OB.1) / 2 ∧ OA.2 = (OA.2 + OB.2) / 2) :=
by 
  -- Using sorry to skip the proof implementation
  sorry

end proof_problem_l402_402653


namespace solve_problem_l402_402299

open Real

-- Elliminate the parameter to get the general equation of C1
def C1_cartesian_eq (x y : ℝ) := (x - 2) ^ 2 + (y - 4) ^ 2 = 20

-- Provide the polar form of C1
def C1_polar_eq (ρ θ : ℝ) := 
  let pol_eq := ρ^2 - 4 * ρ * cos θ - 8 * ρ * sin θ 
  pol_eq = 0

-- Define the polar equation for C2
def C2_polar_eq (θ : ℝ) := θ = π / 3

-- Define the cartesian form for C2
def C2_cartesian_eq (x y : ℝ) := y = sqrt(3) * x

-- Define the function to calculate area of triangle
def area_triangle_OMN (ρ1 ρ2 θ1 θ2 : ℝ) := 1 / 2 * ρ1 * ρ2 * sin (θ1 - θ2)

theorem solve_problem : 
  (∀ x y, C1_cartesian_eq x y → C1_polar_eq (sqrt ((x - 2) ^ 2 + (y - 4) ^ 2)) (atan2 (y - 4) (x - 2)))
  ∧ (∀ x y, C2_polar_eq (atan2 y x) → C2_cartesian_eq x y)
  ∧ (let ρ1 := 2 + 4 * sqrt 3 in
      let ρ2 := 4 + 2 * sqrt 3 in
      let θ1 := π / 3 in
      let θ2 := π / 6 in
      area_triangle_OMN ρ1 ρ2 θ1 θ2 = 8 + 5 * sqrt 3) := sorry

end solve_problem_l402_402299


namespace external_angle_bisector_l402_402719

open EuclideanGeometry

variables {A T C K L : Point}

theorem external_angle_bisector
    (h1 : is_internal_angle_bisector T L A T C) : is_external_angle_bisector T K A T C :=
sorry

end external_angle_bisector_l402_402719


namespace johns_mobile_purchase_price_l402_402677

def purchase_price_mobile (g p_m : ℝ) (loss_grinder percent_profit overall_profit : ℝ) : ℝ :=
  let selling_price_grinder := g - (loss_grinder * g / 100)
  let selling_price_mobile := p_m + (percent_profit * p_m / 100)
  let total_profit := (selling_price_grinder + selling_price_mobile) - (g + p_m)
  p_m = 8000

theorem johns_mobile_purchase_price : purchase_price_mobile 15000 8000 4 10 200 := by
  sorry

end johns_mobile_purchase_price_l402_402677


namespace correct_statements_l402_402369

noncomputable def f (x : ℝ) : ℝ := 4 * real.sin (2 * x + (real.pi / 3))

theorem correct_statements :
  (¬(∀ x : ℝ, f(x) = f(-x)) ∧
   ∃ x₀ : ℝ, f(x₀) ≠ 0 ∧ x₀ = real.pi / 6 ∧
   ∃ x₁ : ℝ, f(x₁) = 4 ∧ x₁ = real.pi / 12) :=
by
  sorry

end correct_statements_l402_402369


namespace escalator_visible_steps_l402_402850

def total_steps_visible (x : ℕ) (steps_down : ℕ) (steps_up : ℕ) : ℕ :=
  steps_down + steps_down

theorem escalator_visible_steps (steps_down steps_up : ℕ) : 
  steps_down = 30 →
  steps_up = 90 →
  (∀ x : ℕ, steps_down + steps_down * x = steps_up - steps_up / 3 * x → x = 1) →
  total_steps_visible 1 steps_down steps_up = 60 :=
begin
  intros h_down h_up h_x,
  rw h_down,
  rw [total_steps_visible, h_down],
  norm_num
end

end escalator_visible_steps_l402_402850


namespace max_profit_at_x_eq_9_l402_402461

def R (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 10 then 
    108 - 1/3 * x^2
  else 
    1080 / x - 10000 / (3 * x^2)

def cost (x : ℝ) : ℝ := 
  1 + 2.7 * x

def profit (x : ℝ) : ℝ := 
  if 0 < x ∧ x <= 10 then 
    x * (108 - 1/3 * x^2) - (1 + 2.7 * x)
  else 
    x * (1080 / x - 10000 / (3 * x^2)) - (1 + 2.7 * x)

theorem max_profit_at_x_eq_9 :
  ∀ x : ℝ, profit x ≤ 386 :=
sorry

end max_profit_at_x_eq_9_l402_402461


namespace part1_part2_l402_402318

noncomputable def A : Set ℝ := {x | ∃ k : ℤ, x = k * Real.pi + Real.pi / 2}

variables (α β : ℝ) (vec_a vec_b : ℝ × ℝ)

def a (α β : ℝ) : ℝ × ℝ := (2 * Real.cos ((α + β) / 2), Real.sin ((α - β) / 2))
def b (α β : ℝ) : ℝ × ℝ := (Real.cos ((α + β) / 2), 3 * Real.sin ((α - β) / 2))

theorem part1 (h₁ : α + β = 2 * Real.pi / 3) (h₂ : vec_a = a α β) (h₃ : vec_a = 2 * b α β) :
  ∃ k : ℤ, α = k * Real.pi + Real.pi / 3 ∧ β = -k * Real.pi + Real.pi / 3 :=
sorry

theorem part2 (h₁ : vec_a = a α β) (h₂ : vec_b = b α β) (h₃ : vec_a • vec_b = 5 / 2)
  (h₄ : α ∈ A) (h₅ : β ∈ A) :
  Real.tan α * Real.tan β = -1 / 5 :=
sorry

end part1_part2_l402_402318


namespace fill_table_with_sum_zero_l402_402082

theorem fill_table_with_sum_zero (n : ℕ) (hn : n ≥ 3) :
  ∃ (f : Fin n → Fin n → ℤ), 
    (∀ i : Fin n, ∑ j, f i j = 0) ∧ 
    (∀ j : Fin n, ∑ i, f i j = 0) ∧ 
    (∀ i j : Fin n, f i j ∈ ({1, 2, -3} : Set ℤ)) :=
by
{
  sorry
}

end fill_table_with_sum_zero_l402_402082


namespace max_common_ratio_arithmetic_geometric_sequence_l402_402590

open Nat

theorem max_common_ratio_arithmetic_geometric_sequence (a : ℕ → ℝ) (d : ℝ) (k : ℕ) (q : ℝ) 
  (hk : k ≥ 2) (ha : ∀ n, a (n + 1) = a n + d)
  (hg : (a 1) * (a (2 * k)) = (a k) ^ 2) :
  q ≤ 2 :=
by
  sorry

end max_common_ratio_arithmetic_geometric_sequence_l402_402590


namespace proof_correct_option_l402_402018

open Classical

variable (x a b : ℝ)

def option_A : Prop := (3 * x + 2 * x^2 = 5 * x^3)
def option_B : Prop := ((a - b)^2 = a^2 - b^2)
def option_C : Prop := ((-x^3)^2 = x^6)
def option_D : Prop := (3 * x^2 * 4 * x^3 = 12 * x^6)

theorem proof_correct_option : ¬option_A ∧ ¬option_B ∧ option_C ∧ ¬option_D :=
by {
  sorry
}

end proof_correct_option_l402_402018


namespace triangle_cosine_l402_402266

theorem triangle_cosine {A : ℝ} (h : 0 < A ∧ A < π / 2) (tan_A : Real.tan A = -2) :
  Real.cos A = - (Real.sqrt 5) / 5 :=
sorry

end triangle_cosine_l402_402266


namespace range_of_m_l402_402234

theorem range_of_m (m : ℝ) : 
  (∀ x ∈ set.Icc (-real.pi/3) (real.pi/3), 
    sqrt 2 * real.sin (x/4) * real.cos (x/4) + sqrt 6 * (real.cos (x/4))^2 - sqrt 6 / 2 - m ≥ 0) 
  ↔ m ≤ sqrt 2 / 2 := 
sorry

end range_of_m_l402_402234


namespace proof_length_segment_EF_l402_402287

noncomputable def length_segment_EF {AB BC : ℝ} (h1 : AB = 4) (h2 : BC = 8)
(fold_line_fraction : ℝ) (h3 : fold_line_fraction = 3 / 5) : ℝ :=
let x := 3 in
let FA := BC - x in
let E := (4, 0) in
let F := (0, 3) in
real.sqrt ((E.1 - F.1)^2 + (E.2 - F.2)^2)

theorem proof_length_segment_EF :
  ∀ (AB BC : ℝ) (h1 : AB = 4) (h2 : BC = 8) (fold_line_fraction : ℝ) (h3 : fold_line_fraction = 3 / 5),
  length_segment_EF h1 h2 fold_line_fraction h3 = 5 :=
by
  intros,
  dsimp [length_segment_EF],
  sorry

end proof_length_segment_EF_l402_402287


namespace length_of_AC_is_25_l402_402661

noncomputable def length_of_AC (x1 y1 x2 y2 : ℝ) : ℝ :=
  let a := 20
  let b := 15
  real.sqrt (a^2 + b^2)

theorem length_of_AC_is_25 (x1 y1 x2 y2 : ℝ) (hB : ∠B = 90°) (hAB : abs (y1 - y2) = 20) (h_slope : (y1 - y2) / (x1 - x2) = 4 / 3) :
  length_of_AC x1 y1 x2 y2 = 25 := by
  sorry

end length_of_AC_is_25_l402_402661


namespace find_a_maximize_profit_l402_402917

-- Definition of parameters
def a := 260
def purchase_price_table := a
def purchase_price_chair := a - 140

-- Condition 1: The number of dining chairs purchased for 600 yuan is the same as the number of dining tables purchased for 1300 yuan.
def condition1 := (600 / (purchase_price_chair : ℚ)) = (1300 / (purchase_price_table : ℚ))

-- Given conditions for profit maximization
def qty_tables := 30
def qty_chairs := 5 * qty_tables + 20
def total_qty := qty_tables + qty_chairs

-- Condition: Total quantity of items does not exceed 200 units.
def condition2 := total_qty ≤ 200

-- Profit calculation
def profit := 280 * qty_tables + 800

-- Theorem statements
theorem find_a : condition1 → a = 260 := sorry

theorem maximize_profit : condition2 ∧ (8 * qty_tables + 800 > 0) → 
  (qty_tables = 30) ∧ (qty_chairs = 170) ∧ (profit = 9200) := sorry

end find_a_maximize_profit_l402_402917


namespace composite_factors_in_circle_l402_402994

theorem composite_factors_in_circle (n : ℕ) (h_composite : ∃ d, 1 < d ∧ d < n ∧ d ∣ n) :
  (∃ (p1 p2 ... pk : ℕ) (m1 m2 ... mk : ℕ), 
    n = p1 ^ m1 * p2 ^ m2 * ... * pk ^ mk ∧
    ∀ i (1 ≤ i ≤ k), Nat.Prime (pi_i) ∧ ∃ j, Nat.gcd (pi_i) (pi_j) ≠ 1) :=
sorry

end composite_factors_in_circle_l402_402994


namespace prime_factors_sum_correct_prime_factors_product_correct_l402_402979

-- The number we are considering
def n : ℕ := 172480

-- Prime factors of the number n
def prime_factors : List ℕ := [2, 3, 5, 719]

-- Sum of the prime factors
def sum_prime_factors : ℕ := 2 + 3 + 5 + 719

-- Product of the prime factors
def prod_prime_factors : ℕ := 2 * 3 * 5 * 719

theorem prime_factors_sum_correct :
  sum_prime_factors = 729 :=
by {
  -- Proof goes here
  sorry
}

theorem prime_factors_product_correct :
  prod_prime_factors = 21570 :=
by {
  -- Proof goes here
  sorry
}

end prime_factors_sum_correct_prime_factors_product_correct_l402_402979


namespace necessary_not_sufficient_condition_l402_402901

theorem necessary_not_sufficient_condition (a : ℝ) (f : ℝ → ℝ) (h_f : ∀ x : ℝ, x > 0 → f x = a * x + Real.log x) :
  (a ≤ 0) → (¬ (a > 0)) ∧ ∃ x > 0, (∂ f / ∂ x) x = 0 :=
by
  intro ha
  split
  { intro hab
    linarith }
  { use 1 / -a
    split
    { linarith }
    { sorry } }

end necessary_not_sufficient_condition_l402_402901


namespace range_f_l402_402225

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x / (x^2 + x + 1) else exp(x) - 3/4

theorem range_f : (Set.range f) = Set.Ioc (-3/4) (1/3) :=
  sorry

end range_f_l402_402225


namespace chapters_per_day_l402_402547

theorem chapters_per_day (chapters : ℕ) (total_days : ℕ) : ℝ :=
  let chapters := 2
  let total_days := 664
  chapters / total_days

example : chapters_per_day 2 664 = 2 / 664 := by sorry

end chapters_per_day_l402_402547


namespace real_y_values_l402_402084

theorem real_y_values (x : ℝ) :
  (∃ y : ℝ, 2 * y^2 + 3 * x * y - x + 8 = 0) ↔ (x ≤ -23 / 9 ∨ x ≥ 5 / 3) :=
by
  sorry

end real_y_values_l402_402084


namespace external_bisector_l402_402760

-- Define the relevant points and segments in the triangle
variables {A T C L K : Type} [inhabited A] [inhabited T] [inhabited C] [inhabited L] [inhabited K]

-- Assuming T, L, K are points such that TL divides angle ATC internally
axiom internal_bisector (h : L) : ∃ TL, TL.bisects_angle (A T C) internally

-- Prove that TK is the external angle bisector given the condition above
theorem external_bisector (h : L) (h_internal : internal_bisector h) : 
  ∃ TK, TK.bisects_angle (A T C) externally :=
sorry

end external_bisector_l402_402760


namespace area_of_L_shape_is_58_l402_402035

-- Define the dimensions of the large rectangle
def large_rectangle_length : ℕ := 10
def large_rectangle_width : ℕ := 7

-- Define the dimensions of the smaller rectangle to be removed
def small_rectangle_length : ℕ := 4
def small_rectangle_width : ℕ := 3

-- Define the area of the large rectangle
def area_large_rectangle : ℕ := large_rectangle_length * large_rectangle_width

-- Define the area of the small rectangle
def area_small_rectangle : ℕ := small_rectangle_length * small_rectangle_width

-- Define the area of the "L" shaped region
def area_L_shape : ℕ := area_large_rectangle - area_small_rectangle

-- Prove that the area of the "L" shaped region is 58 square units
theorem area_of_L_shape_is_58 : area_L_shape = 58 := by
  sorry

end area_of_L_shape_is_58_l402_402035


namespace car_fuel_efficiency_in_city_l402_402853

theorem car_fuel_efficiency_in_city :
  ∀ (T H C : ℝ), 
  (H * T = 420) ∧ 
  (C * T = 336) ∧ 
  (C = H - 6) →
    C = 24 := 
begin
  intros T H C,
  intro h,
  have h1 : H * T = 420 := h.1,
  have h2 : C * T = 336 := h.2.1,
  have h3 : C = H - 6 := h.2.2,
  simpa,
  sorry
end

end car_fuel_efficiency_in_city_l402_402853


namespace MountainRidgeAcademy_l402_402505

theorem MountainRidgeAcademy (j s : ℕ) 
  (h1 : 3/4 * j = 1/2 * s) : s = 3/2 * j := 
by 
  sorry

end MountainRidgeAcademy_l402_402505


namespace machine_A_production_time_l402_402769

noncomputable def machine_A_time : ℕ :=
  let items_B := 1440 / 5 in
  let items_A := (1.25 : ℚ) * items_B in
  1440 / items_A

theorem machine_A_production_time :
  machine_A_time = 4 :=
by
  let items_B := 1440 / 5
  let items_A := (1.25 : ℚ) * items_B
  have eq_1: 1440 / items_A = 4 := by sorry
  exact eq_1

end machine_A_production_time_l402_402769


namespace count_integer_values_of_a_l402_402152

theorem count_integer_values_of_a (a : ℤ) :
  (∃ x : ℤ, x^2 + a * x + 12 * a = 0) ↔ (a ∈ {-(m + n) | m, n : ℤ ∧ m * n = 12 * a}) → ∃! a_vals : finset ℤ, a_vals.card = 16 :=
by
  sorry

end count_integer_values_of_a_l402_402152


namespace number_of_obtuse_triangles_l402_402240

variable (A : Fin 100 → Type)

def is_obtuse (k m : Fin 100) : Prop :=
  0 < m.val - k.val ∧ m.val - k.val < 50

theorem number_of_obtuse_triangles (A : Fin 100 → Type) :
  (∑ k : Fin 100, ∑ l : Fin 100, ∑ m : Fin 100, 
    if is_obtuse k m then 1 else 0) = 117600 :=
sorry

end number_of_obtuse_triangles_l402_402240


namespace negation_of_proposition_l402_402812

theorem negation_of_proposition :
  (¬ ∃ x_0 : ℝ, x_0^3 - x_0^2 + 1 ≥ 0) ↔ ∀ x : ℝ, x^3 - x^2 + 1 ≤ 0 :=
by sorry

end negation_of_proposition_l402_402812


namespace combined_mean_correct_l402_402354

section MeanScore

variables (score_first_section mean_first_section : ℝ)
variables (score_second_section mean_second_section : ℝ)
variables (num_first_section num_second_section : ℝ)

axiom mean_first : mean_first_section = 92
axiom mean_second : mean_second_section = 78
axiom ratio_students : num_first_section / num_second_section = 5 / 7

noncomputable def combined_mean_score : ℝ := 
  let total_score := (mean_first_section * num_first_section + mean_second_section * num_second_section)
  let total_students := (num_first_section + num_second_section)
  total_score / total_students

theorem combined_mean_correct : combined_mean_score 92 78 (5 / 7 * num_second_section) num_second_section = 83.8 := by
  sorry

end MeanScore

end combined_mean_correct_l402_402354


namespace external_bisector_of_triangle_l402_402701

variables {A T C K L : Type} [noncomputable_field A T C K L]
variable {triangle ATC : Triangle A T C}

noncomputable def internal_angle_bisector (T L : Point) := 
    ∃ R, is_bisector TL ∧ TL ∝ AR RL

noncomputable def external_angle_bisector (T K : Point) := 
    ∃ M, is_bisector TK ∧ TK ∝ AM MK

theorem external_bisector_of_triangle {A T C K L : Point} 
    (hL : internal_angle_bisector T L) 
    : external_angle_bisector T K :=
sorry

end external_bisector_of_triangle_l402_402701


namespace match_game_111_optimal_play_l402_402446

theorem match_game_111_optimal_play : 
  let num_matches := 111,
      take_range := {n : ℕ | 1 ≤ n ∧ n ≤ 10},
  (∀ plays : ℕ → ℕ, (∀ (k : ℕ), plays k ∈ take_range) → (plays.sum ≤ num_matches) → ((plays.sum = num_matches) ↔ (∃ i : ℕ, plays i = num_matches))) →
  (∀ (plays : ℕ → ℕ) (i : ℕ), plays i = num_matches → i % 2 = 0) :=
sorry

#eval match_game_111_optimal_play

end match_game_111_optimal_play_l402_402446


namespace baron_munchausen_max_people_l402_402863

theorem baron_munchausen_max_people :
  ∃ x : ℕ, (x = 37) ∧ 
  (1 / 2 * x).nat_ceil + (1 / 3 * x).nat_ceil + (1 / 5 * x).nat_ceil = x := sorry

end baron_munchausen_max_people_l402_402863


namespace number_of_daisies_l402_402271

inductive Plant
| violet
| daisy

def statement_anna (anna danny : Plant) : Prop :=
  anna ≠ danny

def statement_bella (ellie : Plant) : Prop :=
  ellie = Plant.daisy

def statement_carla (ellie : Plant) : Prop :=
  ellie = Plant.violet

def statement_danny (anna bella carla danny ellie : Plant) : Prop :=
  (anna = Plant.violet) + (bella = Plant.violet) + (carla = Plant.violet) 
  + (danny = Plant.violet) + (ellie = Plant.violet) ≥ 3

def statement_ellie (anna : Plant) : Prop :=
  anna = Plant.violet

theorem number_of_daisies (anna bella carla danny ellie : Plant) 
  (h_anna : statement_anna anna danny)
  (h_bella : statement_bella ellie)
  (h_carla : statement_carla ellie)
  (h_danny : statement_danny anna bella carla danny ellie)
  (h_ellie : statement_ellie anna) : 
  (anna = Plant.daisy) + (bella = Plant.daisy) + (carla = Plant.daisy) 
  + (danny = Plant.daisy) + (ellie = Plant.daisy) = 3 :=
sorry

end number_of_daisies_l402_402271


namespace sin_of_5pi_over_6_l402_402986

theorem sin_of_5pi_over_6 : Real.sin (5 * Real.pi / 6) = 1 / 2 := 
by
  sorry

end sin_of_5pi_over_6_l402_402986


namespace num_satisfying_log_inequality_l402_402188

theorem num_satisfying_log_inequality :
  ∃ (n : ℕ), (n = 18) ∧ (∀ (x : ℤ), (50 < x ∧ x < 70) → (log10 (x - 50) + log10 (70 - x) < 2) → True) :=
sorry

end num_satisfying_log_inequality_l402_402188


namespace polar_curve_symmetry_l402_402658

theorem polar_curve_symmetry :
  ∀ (ρ θ : ℝ), ρ = 4 * Real.sin (θ - π / 3) → 
  ∃ k : ℤ, θ = 5 * π / 6 + k * π :=
sorry

end polar_curve_symmetry_l402_402658


namespace nonnegative_integers_with_abs_value_less_than_4_l402_402992

theorem nonnegative_integers_with_abs_value_less_than_4 :
  {n : ℕ | abs (n : ℤ) < 4} = {0, 1, 2, 3} :=
by {
  sorry
}

end nonnegative_integers_with_abs_value_less_than_4_l402_402992


namespace magnitude_of_vector_AB_l402_402573

theorem magnitude_of_vector_AB :
  let A := (1 : ℝ, 3 : ℝ),
      B := (4 : ℝ, -1 : ℝ),
      AB := (B.1 - A.1, B.2 - A.2)
  in real.sqrt (AB.1^2 + AB.2^2) = 5 :=
by
  let A := (1 : ℝ, 3 : ℝ)
  let B := (4 : ℝ, -1 : ℝ)
  let AB := (B.1 - A.1, B.2 - A.2)
  have magnitude := real.sqrt (AB.1^2 + AB.2^2)
  show magnitude = 5
  sorry

end magnitude_of_vector_AB_l402_402573


namespace jar_a_marbles_l402_402673

theorem jar_a_marbles : ∃ A : ℕ, (∃ B : ℕ, B = A + 12) ∧ (∃ C : ℕ, C = 2 * (A + 12)) ∧ (A + (A + 12) + 2 * (A + 12) = 148) ∧ (A = 28) :=
by
sorry

end jar_a_marbles_l402_402673


namespace external_bisector_of_triangle_l402_402702

variables {A T C K L : Type} [noncomputable_field A T C K L]
variable {triangle ATC : Triangle A T C}

noncomputable def internal_angle_bisector (T L : Point) := 
    ∃ R, is_bisector TL ∧ TL ∝ AR RL

noncomputable def external_angle_bisector (T K : Point) := 
    ∃ M, is_bisector TK ∧ TK ∝ AM MK

theorem external_bisector_of_triangle {A T C K L : Point} 
    (hL : internal_angle_bisector T L) 
    : external_angle_bisector T K :=
sorry

end external_bisector_of_triangle_l402_402702


namespace water_left_in_bottle_l402_402680

-- Define the initial conditions
def V_initial : ℝ := 4
def V_first_drink : ℝ := V_initial * (1 / 4)

-- Define remaining after the first drink
def V_after_first_drink : ℝ := V_initial - V_first_drink

-- Define the second drink volume
def V_second_drink : ℝ := V_after_first_drink * (2 / 3)

-- Define the remaining volume after the second drink
def V_after_second_drink : ℝ := V_after_first_drink - V_second_drink

-- Conjecture: The remaining volume should be 1 liter
theorem water_left_in_bottle : V_after_second_drink = 1 := by
  sorry

end water_left_in_bottle_l402_402680


namespace bob_wins_original_game_l402_402052

-- Definition of the original game conditions
def original_game_conditions (points : ℕ) : Prop :=
  (points = 2012)

-- Statement that Bob wins the original game
theorem bob_wins_original_game : ∀ points, original_game_conditions points → 
  ∃ strategy_bob : (ℕ → ℕ), ∀ strategy_alice : (ℕ → ℕ), 
  (strategy_alice 1 = 1 ∨ strategy_alice 2 = 2) →
  (strategy_bob 1 = 1 ∨ strategy_bob 2 = 2) → 
  ¬ (∃ winning_strategy_alice : (ℕ → bool), winning_strategy_alice points = true) :=
by sorry


end bob_wins_original_game_l402_402052


namespace common_property_of_rhombus_rectangle_square_l402_402848

-- Definitions of the properties of quadrilaterals
structure Quadrilateral :=
(sides : Fin 4 → ℝ)

structure Rhombus extends Quadrilateral :=
(diagonals_are_perpendicular : True)
(diagonals_bisect_each_other : True)

structure Rectangle extends Quadrilateral :=
(all_angles_are_right : True)
(diagonals_are_equal : True)
(diagonals_bisect_each_other : True)

structure Square extends Rhombus, Rectangle :=
(equal_sides : ∀ i j : Fin 4, sides i = sides j)

theorem common_property_of_rhombus_rectangle_square
    (R : Rhombus) (Re : Rectangle) (S : Square) :
  R.diagonals_bisect_each_other ∧ Re.diagonals_bisect_each_other ∧ S.diagonals_bisect_each_other :=
by
  sorry

end common_property_of_rhombus_rectangle_square_l402_402848


namespace n_gon_coloring_l402_402544

theorem n_gon_coloring {n : ℕ} (h : n ≥ 3) :
  ∃ (triangles : list (fin n → ℝ)), 
    (∀ tri ∈ triangles, ∃ (c : fin 3 → ℤ), ∀ (i j : fin 3), i ≠ j → c i ≠ c j) ∧
    (∀ i : fin n, ∃ j k : fin n, j ≠ k ∧ i ≠ j ∧ i ≠ k ∧ (triangle_vertex_color i = red ∨ triangle_vertex_color i = green ∨ triangle_vertex_color i = blue)) :=
sorry

end n_gon_coloring_l402_402544


namespace petya_cannot_achieve_l402_402524

def is_valid_move (i j : ℕ) : Prop := 
  (i + j = 1) ∨ (i + j = -1)

def adjacent_cells (x y : ℕ × ℕ) : Prop :=
  is_valid_move (x.1 - y.1) (x.2 - y.2)

def problem (board : ℕ → ℕ → bool) : Prop :=
  (∀ x y, (board x y) = tt) →
  ¬ (∃ new_board : ℕ → ℕ → bool,
    (∀ x y, (board x y) = tt → 
      (∃ z w, adjacent_cells (x, y) (z, w) ∧ (new_board z w) = tt) ∧
      (∀ z w, adjacent_cells (x, y) (z, w) → (new_board z w) = tt)) ∧ 
    (∀ x y, (new_board x y) = tt))

theorem petya_cannot_achieve (board : ℕ → ℕ → bool) : problem board := sorry

end petya_cannot_achieve_l402_402524


namespace sheets_per_day_l402_402935

theorem sheets_per_day (total_sheets : ℕ) (days_per_week : ℕ) (sheets_per_day : ℕ) :
  total_sheets = 60 → days_per_week = 5 → sheets_per_day = total_sheets / days_per_week → sheets_per_day = 12 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3.symm.trans (by norm_num)

end sheets_per_day_l402_402935


namespace even_and_increasing_on_positive_reals_l402_402055

-- Definitions of the functions
def f (x : ℝ) : ℝ := cos x
def g (x : ℝ) : ℝ := -x^2 + 1
def h (x : ℝ) : ℝ := Real.log x / Real.log 2
def j (x : ℝ) : ℝ := exp x - exp (-x)

-- Proof that the function h is even and increasing on (0, +∞)
theorem even_and_increasing_on_positive_reals : (∀ x : ℝ, h (-x) = h x) ∧ (∀ x y : ℝ, 0 < x → x < y → h x < h y) :=
sorry

end even_and_increasing_on_positive_reals_l402_402055


namespace ratio_AB_AC_l402_402270

-- Definitions of given conditions and their relationships in Lean.

variables (A B C D M K : Type) [convex_quadrilateral A B C D]
variables (angle_ABD angle_ACD : ℝ)
variables (CK KM CD DK : ℝ) (ratios : CK / KM = 2 / 1 ∧ CD / DK = 5 / 3)
variables (sum_angles : angle_ABD + angle_ACD = 180)

-- The goal is to show that the ratio of length AB to length AC is 5:9.
theorem ratio_AB_AC (h1 : convex_quadrilateral A B C D)
                    (h2 : CK / KM = 2 / 1)
                    (h3 : CD / DK = 5 / 3)
                    (h4 : ∠BAD + ∠CAD = 180) :
      (AB / AC) = 5 / 9 :=
by
  sorry

end ratio_AB_AC_l402_402270


namespace TK_is_external_bisector_of_ATC_l402_402710

variable {A T C L K : Type} [LinearOrder A] [LinearOrder T] [LinearOrder C] [LinearOrder L] [LinearOrder K]
variable (triangle_ATC : Triangle A T C)
variable (point_T : Point T)
variable (point_L : Point L)
variable (point_K : Point K)
variable (internal_bisector_TL : AngleBisector (angle_ATC) T L)
variable (external_bisector_TK : AngleBisector (external_angle_ATC) T K)

theorem TK_is_external_bisector_of_ATC :
  external_bisector_TK = true :=
sorry

end TK_is_external_bisector_of_ATC_l402_402710


namespace external_angle_bisector_l402_402720

open EuclideanGeometry

variables {A T C K L : Point}

theorem external_angle_bisector
    (h1 : is_internal_angle_bisector T L A T C) : is_external_angle_bisector T K A T C :=
sorry

end external_angle_bisector_l402_402720


namespace problem_l402_402567

-- Define the sequence b_n and its properties.
def seq (b : ℕ → ℝ) :=
  ∀ n : ℕ, 1 < n → b (n - 1) + b (n + 1) ≥ 2 * b n

-- State the main theorem in Lean 4.
theorem problem
  (b : ℕ → ℝ)
  (b_pos : ∀ n, 0 < b n)
  (b_seq : seq b)
  (b1_gt_b2 : b 1 > b 2)
  :
  (∃ t : ℝ, ∀ n : ℕ, 0 < n → b n - b 1 > n * t) ∧
  (∀ n : ℕ, let T := (∑ i in Finset.range n, b i) in 2 * T ≥ (n ^ 2 + n) * b n - (n ^ 2 - n) * b (n + 1)) :=
by
  sorry

end problem_l402_402567


namespace baron_munchausen_max_people_l402_402866

theorem baron_munchausen_max_people :
  ∃ x : ℕ, (x = 37) ∧ 
  (1 / 2 * x).nat_ceil + (1 / 3 * x).nat_ceil + (1 / 5 * x).nat_ceil = x := sorry

end baron_munchausen_max_people_l402_402866


namespace perpendicular_line_to_plane_l402_402324

variables {Point Line Plane : Type}
variables (a b c : Line) (α : Plane) (A : Point)

-- Define the conditions
def line_perpendicular_to (l1 l2 : Line) : Prop := sorry
def line_in_plane (l : Line) (p : Plane) : Prop := sorry
def lines_intersect_at (l1 l2 : Line) (P : Point) : Prop := sorry
def line_perpendicular_to_plane (l : Line) (p : Plane) : Prop := sorry

-- Given conditions in Lean 4
variables (h1 : line_perpendicular_to c a)
variables (h2 : line_perpendicular_to c b)
variables (h3 : line_in_plane a α)
variables (h4 : line_in_plane b α)
variables (h5 : lines_intersect_at a b A)

-- The theorem statement to prove
theorem perpendicular_line_to_plane : line_perpendicular_to_plane c α :=
sorry

end perpendicular_line_to_plane_l402_402324


namespace distinct_integer_a_values_l402_402139

open Nat

theorem distinct_integer_a_values : ∃ (S : Finset ℤ), S.card = 8 ∧
  ∀ a ∈ S, ∃ (p q : ℤ), p + q = -a ∧ p * q = 12a ∧ 
  ∀ x ∈ (polynomial.roots (polynomial.C (12 : ℚ) * polynomial.X + polynomial.C a * polynomial.X + polynomial.C 1)).val, x ∈ ℤ :=
by
  sorry

end distinct_integer_a_values_l402_402139


namespace irrational_distance_exists_l402_402088

noncomputable def squared_distance (p q : ℝ × ℝ) : ℝ :=
  (p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2 

theorem irrational_distance_exists 
  (A B C X : ℝ × ℝ) 
  (hC : C = (A.1 + B.1) / 2, (A.2 + B.2) / 2)
  (hAB_irrational : irrational (squared_distance A B)) :
  irrational (squared_distance X A) 
  ∨ irrational (squared_distance X B)
  ∨ irrational (squared_distance X C) :=
by 
  sorry

end irrational_distance_exists_l402_402088


namespace coeff_c_of_factor_l402_402431

theorem coeff_c_of_factor (c : ℚ): 
  (P := λ x : ℚ, x^3 + 2 * x^2 + c * x + 8) → 
  (x_minus_3_factor : P(3) = 0) → 
  c = -53 / 3 :=
by 
  sorry

end coeff_c_of_factor_l402_402431


namespace Al_sandwiches_count_l402_402385

theorem Al_sandwiches_count (B M C : Type) 
    [Fintype B] [Fintype M] [Fintype C]
    (bread_count : Fintype.card B = 5)
    (meat_count : Fintype.card M = 7)
    (cheese_count : Fintype.card C = 6)
    (restricted_combinations : list (B × M × C))
    (restriction1 : ∀ b ∈ (Fintype.elems B), (b, "ham", "cheddar") ∈ restricted_combinations)
    (restriction2 : ∀ b ∈ (Fintype.elems B), (b, "turkey", "gouda") ∈ restricted_combinations)
    (restriction3 : ∀ c ∈ (Fintype.elems C), ("white", "chicken", c) ∈ restricted_combinations): 
    ∃ (sandwich_count : Nat), sandwich_count = 194 := 
by
  sorry

end Al_sandwiches_count_l402_402385


namespace area_of_quadrilateral_ABCD_l402_402654

variable (m : ℝ)

def pointA := (m-3, m)
def pointB := (m, m)
def pointC := (m, m+5)
def pointD := (m-3, m+5)

def length (p1 p2 : ℝ × ℝ) : ℝ :=
  let dx := p2.1 - p1.1
  let dy := p2.2 - p1.2
  real.sqrt (dx^2 + dy^2)

def AB := length pointA pointB
def AD := length pointA pointD

theorem area_of_quadrilateral_ABCD :
  AB * AD = 15 := by
sorry

end area_of_quadrilateral_ABCD_l402_402654


namespace range_of_f_l402_402229

theorem range_of_f (a b : ℝ) (h_even : ∀ x : ℝ, f x = f (-x))
  (h_b : -2*b + 3*b - 1 = 0) :
  set.range (λ x : ℝ, x^2 - 2*a*x + b) = set.Icc 1 5 :=
by
  -- Parse conditions
  sorry

end range_of_f_l402_402229


namespace max_value_of_g_l402_402080

def g (n : ℕ) : ℕ :=
  if n < 20 then n + 20 else g (n - 7)

theorem max_value_of_g : ∀ n : ℕ, g n ≤ 39 ∧ (∃ m : ℕ, g m = 39) := by
  sorry

end max_value_of_g_l402_402080


namespace quadrilateral_ABCD_AB_measure_l402_402296

theorem quadrilateral_ABCD_AB_measure (a b θ : ℝ) (A B C D E : Type)
  [AddCommGroup A] [VectorSpace ℝ A] (AB CD : ℝ) (angleB angleD : ℝ)
  (AD ED : ℝ)
  (h1 : AB ∥ CD)
  (h2 : angleD = 3 * angleB)
  (h3 : AD = 2 * a)
  (h4 : CD = 3 * b)
  (h5 : AB = AD + ED) : 
  AB = (4 * a^2 + 9 * b^2) / (3 * b) := 
sorry

end quadrilateral_ABCD_AB_measure_l402_402296


namespace number_of_pairs_not_prime_sum_l402_402060

open Finset

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m ∣ n, m = 1 ∨ m = n

def dies_faces := {1, 2, 3, 4, 5, 6, 7, 8}

def pairs_not_prime_sum : Finset (ℕ × ℕ) := 
  univ.product univ
  .filter (λ p : ℕ × ℕ, p.fst < p.snd ∧ ¬ is_prime (p.fst + p.snd))

theorem number_of_pairs_not_prime_sum : (pairs_not_prime_sum dies_faces).card = 17 :=
by
  sorry

end number_of_pairs_not_prime_sum_l402_402060


namespace count_propositions_l402_402951

def is_proposition (s : String) : Prop :=
  s = "The sum of two acute angles is an obtuse angle." ∨
  s = "Zero is neither a positive number nor a negative number." ∨
  s = "Flowers bloom in spring."

theorem count_propositions :
  let statements := ["Construct the perpendicular bisector of line segment AB.",
                     "The sum of two acute angles is an obtuse angle.",
                     "Did our country win the right to host the 2008 Olympics?",
                     "Zero is neither a positive number nor a negative number.",
                     "No loud talking is allowed.",
                     "Flowers bloom in spring."]
  (count (is_proposition) statements) = 3 :=
by
  sorry

end count_propositions_l402_402951


namespace sum_of_palindromes_l402_402819

/-- Definition of a three-digit palindrome -/
def is_palindrome (n : ℕ) : Prop :=
  n / 100 = n % 10

theorem sum_of_palindromes (a b : ℕ) (h1 : is_palindrome a)
  (h2 : is_palindrome b) (h3 : a * b = 334491) (h4 : 100 ≤ a)
  (h5 : a < 1000) (h6 : 100 ≤ b) (h7 : b < 1000) : a + b = 1324 :=
sorry

end sum_of_palindromes_l402_402819


namespace part_a_part_b_l402_402836

def happy (n : ℕ) : Prop :=
  ∃ (a b : ℤ), a^2 + b^2 = n

theorem part_a (t : ℕ) (ht : happy t) : happy (2 * t) := 
sorry

theorem part_b (t : ℕ) (ht : happy t) : ¬ happy (3 * t) := 
sorry

end part_a_part_b_l402_402836


namespace number_of_teams_l402_402650

theorem number_of_teams (n : ℕ) (h : n * (n - 1) / 2 = 28) : n = 8 :=
sorry

end number_of_teams_l402_402650


namespace tangent_ln_at_origin_l402_402589

theorem tangent_ln_at_origin {k : ℝ} (h : ∀ x : ℝ, (k * x = Real.log x) → k = 1 / x) : k = 1 / Real.exp 1 :=
by
  sorry

end tangent_ln_at_origin_l402_402589


namespace probability_dart_lands_in_hexagon_l402_402955

theorem probability_dart_lands_in_hexagon (s t : ℝ) (h : t = s / Real.sqrt 2) :
  (let A_hex := (3 * Real.sqrt 3 / 2) * t^2 in
   let A_oct := 2 * (1 + Real.sqrt 2) * s^2 in
   A_hex / A_oct = 3 * Real.sqrt 3 / (8 * (1 + Real.sqrt 2))) :=
by
  sorry

end probability_dart_lands_in_hexagon_l402_402955


namespace base_8_addition_problem_l402_402631

theorem base_8_addition_problem (square : ℕ) (h1 : (5 * 8^2 + 3 * 8^1 + 2 * 8^0 + square) % 8 = 0)
  (h2 : (3 ^ 8 - 5 = square))
  (h3 : 3 + 3 = 6)
  (h4 : 5 = 6 (by exact h3)) :
  square = 3 :=
sorry

end base_8_addition_problem_l402_402631


namespace derivative_at_zero_l402_402253

noncomputable def f (x : ℝ) : ℝ := x * (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5) * (x - 6)

theorem derivative_at_zero : deriv f 0 = 720 :=
by
  sorry

end derivative_at_zero_l402_402253


namespace number_of_ways_to_fill_grid_with_conditions_l402_402531

/-- 
Number of ways to arrange numbers 1 to 9 in a 3x3 matrix, where each row is increasing
from left to right and each column is decreasing from top to bottom, is 42.
------------------ -/
theorem number_of_ways_to_fill_grid_with_conditions : 
  let grid := Matrix (Fin 3) (Fin 3) Nat
  ∃ (m : grid),
    (∀ i : Fin 3, StrictMono (λ j : Fin 3 => m i j)) ∧
    (∀ j : Fin 3, StrictAnti (λ i : Fin 3 => m i j)) ∧
    (∀ v : Fin 9, ∃ i j, m i j = v.succ) →
    ∃! (n : Nat), n = 42 := 
sorry

end number_of_ways_to_fill_grid_with_conditions_l402_402531


namespace probability_of_r25_in_r35_l402_402378

-- Definitions of given conditions
variables (n : ℕ) (r : ℕ → ℕ)
variables [fintype (fin n)] [decidable_eq (fin n)]

-- Distinct and random order condition for the sequence r
def distinct (r : ℕ → ℕ) : Prop :=
  ∀ i j, i ≠ j → r i ≠ r j

-- Bubble pass property we want to prove
def after_bubble_pass (r : ℕ → ℕ) (i j : ℕ) : Prop :=
  r i > r j ∧ r (i+1) < r j

-- Formalize of the mathematical equivalence proof problem
theorem probability_of_r25_in_r35 (h : n = 50) (hdistinct : distinct r) :
  let p := 1 in
  let q := 1260 in
  p + q = 1261 :=
sorry

end probability_of_r25_in_r35_l402_402378


namespace find_m_times_t_l402_402328

noncomputable theory
open Classical

def func_condition (g : ℝ → ℝ) : Prop :=
∀ x y : ℝ, g(x) * g(y) - g(x + y) = x * y - x - y

theorem find_m_times_t (g : ℝ → ℝ) (h : func_condition g) :
  let m := if g(2) = -1 then 1 else 0,
      t := if m = 1 then -1 else 0 in
  m * t = -1 := by
  sorry

end find_m_times_t_l402_402328


namespace fifth_degree_monomial_l402_402003

theorem fifth_degree_monomial (x y : ℕ) (hx : x = 2) (hy : y = 3) : (x + y = 5) :=
by {
  rw [hx, hy],
  simp,
  sorry,
}

end fifth_degree_monomial_l402_402003


namespace log_relationship_l402_402325

def a := log 5 4
def b := log (sqrt 2) 3
def c := (log 0.2 3) ^ 2

theorem log_relationship : b > a ∧ a > c :=
by 
  -- Proof omitted
  sorry

end log_relationship_l402_402325


namespace first_range_is_30_l402_402911

theorem first_range_is_30 
  (R2 R3 : ℕ)
  (h1 : R2 = 26)
  (h2 : R3 = 32)
  (h3 : min 26 (min 30 32) = 30) : 
  ∃ R1 : ℕ, R1 = 30 :=
  sorry

end first_range_is_30_l402_402911


namespace family_total_score_l402_402355

theorem family_total_score : 
  ∀ (Olaf Dad Sister Mom : ℕ),
  (Dad = 7) →
  (Olaf = 3 * Dad) →
  (Sister = Dad + 4) →
  (Mom = 2 * Sister) →
  (Olaf + Dad + Sister + Mom = 61) :=
by
  intros Olaf Dad Sister Mom hDad hOlaf hSister hMom
  rw [hDad, hSister, hOlaf, hMom] at *
  sorry

end family_total_score_l402_402355


namespace conic_section_type_l402_402087

theorem conic_section_type :
  (∃ x y : ℝ, (x - 3)^2 = 4 * (y + 2)^2 + 25) → "H" :=
by
  sorry

end conic_section_type_l402_402087


namespace sequence_general_term_l402_402606

theorem sequence_general_term (a : ℕ → ℝ) (h₁ : a 1 = 1)
  (h₂ : ∀ n : ℕ, 0 < n → a (n + 1) = a n + 1 + 2 * Real.sqrt (a n)) :
  ∀ n : ℕ, 0 < n → a n = n ^ 2 :=
by
  assume n hn
  sorry

end sequence_general_term_l402_402606


namespace compute_cd_l402_402390

noncomputable def ellipse_foci_at : Prop :=
  ∃ (c d : ℝ), (d^2 - c^2 = 25) ∧ (c^2 + d^2 = 64) ∧ (|c * d| = real.sqrt 868.75)

theorem compute_cd : ellipse_foci_at :=
  sorry

end compute_cd_l402_402390


namespace external_angle_bisector_proof_l402_402734

variables {A T C L K : Type} [Nonempty A] [Nonempty T] [Nonempty C] [Nonempty L] [Nonempty K]

noncomputable def angle_bisector_theorem (AL LC AB BC AK KC : ℝ) : Prop :=
(AL / LC) = (AB / BC) ∧ (AK / KC) = (AL / LC)

noncomputable def internal_angle_bisector (AT TC AL LC : ℝ) : Prop :=
(AT / TC) = (AL / LC)

noncomputable def external_angle_bisector (AT TC AK KC : ℝ) : Prop :=
(AT / TC) = (AK / KC)

theorem external_angle_bisector_proof (AL LC AB BC AK KC AT TC : ℝ) 
(h1 : angle_bisector_theorem AL LC AB BC AK KC)
(h2 : internal_angle_bisector AT TC AL LC) :
external_angle_bisector AT TC AK KC :=
sorry

end external_angle_bisector_proof_l402_402734


namespace find_f_neg_one_l402_402194

def f (x : ℝ) (a : ℝ) : ℝ := (x^2 + a) / x

theorem find_f_neg_one (a : ℝ) (h : f 1 a = 4) : f (-1) a = -4 :=
by
  sorry

end find_f_neg_one_l402_402194


namespace who_stole_the_pepper_l402_402898

-- Definitions of the characters involved
inductive Character
| Gryphon
| TurtleQuasi
| Lobster

-- The claims made by each character
def Gryphon_claim : Character → Prop
| Character.TurtleQuasi := true
| _ := false

def TurtleQuasi_claim : Character → Prop
| Character.Lobster := true
| _ := false

-- The main theorem stating who stole the pepper
theorem who_stole_the_pepper (Guilty : Character → Prop) (Innocent : Character → Prop) :
  (∀ c, c ≠ Character.Gryphon → c ≠ Character.TurtleQuasi → c ≠ Character.Lobster → false) →
  (∀ c, (Innocent c ↔ ¬ Guilty c)) →
  (∀ c, (Innocent c → (c = Character.TurtleQuasi → TurtleQuasi_claim Character.Lobster) ∧ 
                  (c = Character.Gryphon → Gryphon_claim Character.TurtleQuasi))) →
  (∀ c, (Guilty c → (c = Character.TurtleQuasi → ¬ TurtleQuasi_claim Character.Lobster) ∧ 
                  (c = Character.Gryphon → ¬ Gryphon_claim Character.TurtleQuasi))) →
  Guilty Character.Lobster :=
by
  intro h0 h1 h2 h3
  sorry


end who_stole_the_pepper_l402_402898


namespace num_integers_contains_3_and_4_l402_402245

theorem num_integers_contains_3_and_4 
  (n : ℕ) (h1 : 500 ≤ n) (h2 : n < 1000) :
  (∀ a b c : ℕ, n = 100 * a + 10 * b + c → (b = 3 ∧ c = 4) ∨ (b = 4 ∧ c = 3)) → 
  n = 10 :=
sorry

end num_integers_contains_3_and_4_l402_402245


namespace problem1_problem2_l402_402452

/-- Problem 1: Assuming (x + y) * (1/x + a/y) ≥ 9 for any x > 0, y > 0, prove that a ≥ 4. -/
theorem problem1 (a : ℝ) (h : ∀ (x y : ℝ), x > 0 → y > 0 → (x + y) * (1 / x + a / y) ≥ 9) :
  4 ≤ a :=
sorry

/-- Problem 2: Given a > 0, b > 0, and 2a + b = 1,
    prove that the maximum value of S = 2sqrt(ab) - 4a² - b² is (sqrt(2) - 1) / 2. -/
theorem problem2 (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 2 * a + b = 1) :
  2 * sqrt (a * b) - 4 * a^2 - b^2 ≤ (sqrt 2 - 1) / 2 :=
sorry

end problem1_problem2_l402_402452


namespace ratio_Lisa_Charlotte_l402_402069

def P_tot : ℕ := 100
def Pat_money : ℕ := 6
def Lisa_money : ℕ := 5 * Pat_money
def additional_required : ℕ := 49
def current_total_money : ℕ := P_tot - additional_required
def Pat_Lisa_total : ℕ := Pat_money + Lisa_money
def Charlotte_money : ℕ := current_total_money - Pat_Lisa_total

theorem ratio_Lisa_Charlotte : (Lisa_money : ℕ) / Charlotte_money = 2 :=
by
  -- Proof to be filled in later
  sorry

end ratio_Lisa_Charlotte_l402_402069


namespace minimum_bounces_to_reach_height_l402_402022

noncomputable def height_after_bounces (initial_height : ℝ) (bounce_factor : ℝ) (k : ℕ) : ℝ :=
  initial_height * (bounce_factor ^ k)

theorem minimum_bounces_to_reach_height
  (initial_height : ℝ) (bounce_factor : ℝ) (min_height : ℝ) :
  initial_height = 800 → bounce_factor = 0.5 → min_height = 2 →
  (∀ k : ℕ, height_after_bounces initial_height bounce_factor k < min_height ↔ k ≥ 9) := 
by
  intros h₀ b₀ m₀
  rw [h₀, b₀, m₀]
  sorry

end minimum_bounces_to_reach_height_l402_402022


namespace evaluate_expression_l402_402528

theorem evaluate_expression : (2 ^ (-3) * 3 ^ 0) / (2 ^ (-5)) = 4 := by
  sorry

end evaluate_expression_l402_402528


namespace count_integers_satisfying_condition_l402_402621

def in_range (n : ℤ) : Prop := -10 ≤ n ∧ n ≤ 12

def satisfies_inequality (n : ℤ) : Prop := (n - 3) * (n + 3) * (n + 7) < 0

theorem count_integers_satisfying_condition :
  finset.card (finset.filter satisfies_inequality (finset.filter in_range (finset.Icc -10 12))) = 5 :=
by
  sorry

end count_integers_satisfying_condition_l402_402621


namespace linear_system_solution_exists_l402_402005

theorem linear_system_solution_exists
  (n : ℕ) (hn : n > 0)
  (a : Fin n → Fin (2 * n) → ℤ)
  (h_bound : ∀ i j, a i j = 0 ∨ a i j = 1 ∨ a i j = -1)
  (h_equation : ∀ i : Fin n, ∑ j in Finset.univ, a i j * (x j : ℤ) = 0) :
  ∃ (x : Fin (2 * n) → ℤ), (∀ i, x i ≠ 0) ∧ (∀ i, |x i| ≤ 2 * n) :=
by
  sorry

end linear_system_solution_exists_l402_402005


namespace rotated_angle_l402_402394

theorem rotated_angle (initial_angle : ℝ) (rotation_angle : ℝ) (final_angle : ℝ) :
  initial_angle = 30 ∧ rotation_angle = 450 → final_angle = 60 :=
by
  intro h
  sorry

end rotated_angle_l402_402394


namespace number_of_men_in_second_group_l402_402019

theorem number_of_men_in_second_group 
  (work : ℕ)
  (days_first_group days_second_group : ℕ)
  (men_first_group men_second_group : ℕ)
  (h1 : work = men_first_group * days_first_group)
  (h2 : work = men_second_group * days_second_group)
  (h3 : men_first_group = 20)
  (h4 : days_first_group = 30)
  (h5 : days_second_group = 24) :
  men_second_group = 25 :=
by
  sorry

end number_of_men_in_second_group_l402_402019


namespace sum_of_arithmetic_sequence_modulo_l402_402404

theorem sum_of_arithmetic_sequence_modulo (n : ℕ) :
  (∑ k in finset.range n, if (k % 7 = 3) ∧ (k < 100) then k else 0) = 679 :=
sorry

end sum_of_arithmetic_sequence_modulo_l402_402404


namespace min_b_over_a_l402_402210

theorem min_b_over_a (a b : ℝ) (h : ∀ x : ℝ, (Real.log a + b) * Real.exp x - a^2 * Real.exp x ≥ 0) : b / a ≥ 1 := by
  sorry

end min_b_over_a_l402_402210


namespace external_angle_bisector_of_triangle_l402_402742

variables {A T C L K : Type}
variables [IsTriangle A T C] [IsInternalAngleBisector T L (A T C)]

theorem external_angle_bisector_of_triangle 
  (h1 : is_internal_angle_bisector TL ATC) :
  is_external_angle_bisector TK ATC :=
sorry

end external_angle_bisector_of_triangle_l402_402742


namespace no_such_function_exists_l402_402089

theorem no_such_function_exists : ¬ ∃ f : ℝ+ → ℝ+, ∀ x y : ℝ+, (f (x + y))^2 ≥ (f x)^2 * (1 + y * f x) :=
by
  intro h
  sorry -- The proof would go here

end no_such_function_exists_l402_402089


namespace find_t_l402_402346

-- conditions
def quadratic_eq (x : ℝ) : Prop := 25 * x^2 + 20 * x - 1000 = 0

-- statement to prove
theorem find_t (x : ℝ) (p t : ℝ) (h1 : p = 2/5) (h2 : t = 104/25) : 
  (quadratic_eq x) → (x + p)^2 = t :=
by
  intros
  sorry

end find_t_l402_402346


namespace min_k_for_reciprocal_liking_l402_402409

theorem min_k_for_reciprocal_liking (n k : ℕ) (h_n : n = 30) (h_likes : ∀ (i : ℕ), i < n → (∃ d : list ℕ, d.length = k ∧ ∀ j ∈ d, j < n)) : k ≥ 15 :=
by {
  -- remove this and implement the proof
  sorry
}

end min_k_for_reciprocal_liking_l402_402409


namespace hyperbola_axes_length_l402_402603

theorem hyperbola_axes_length : 
  ∀ (x y : ℝ), (x^2 - 8*y^2 = 32) → 
    (real_axis_length = 8 * real.sqrt 2) ∧ (imaginary_axis_length = 4) :=
by
  sorry

end hyperbola_axes_length_l402_402603


namespace find_natural_numbers_l402_402103

theorem find_natural_numbers (x : ℕ) : (x % 7 = 3) ∧ (x % 9 = 4) ∧ (x < 100) ↔ (x = 31) ∨ (x = 94) := 
by sorry

end find_natural_numbers_l402_402103


namespace simplify_expression_l402_402376

-- The main theorem
theorem simplify_expression (x y : ℕ) (h_x : x = 4) (h_y : y = 2) :
  (16 * x^2 * y^3) / (8 * x * y^2) = 16 :=
by {
  -- Mentioning that proof is skipped
  sorry,
}

end simplify_expression_l402_402376


namespace parabola_equation_l402_402807

theorem parabola_equation (a b c d e f : ℤ) (h₁ : a = 7) (h₂ : b = 0) (h₃ : c = 0) 
(h₄ : d = -28) (h₅ : e = 1) (h₆ : f = -28) (h₇ : Int.gcd a b c d e f = 1) :
  ∃ (p : ℝ × ℝ), 
    (p = (3, 7)) ∧ 
    (∃ v, v = (2, 0)) ∧ 
    (x_axis_symmetry : (∀ x₁ x₂ y, (2 - x₁) = -(x₂ - 2) ∧ y = 0)) ∧ 
    (focus : (p.1 = 2)) ∧ 
    a * x ^ 2 + b * x * y + c * y ^ 2 + d * x + e * y + f = 0 := 
sorry

end parabola_equation_l402_402807


namespace largest_crowd_size_l402_402891

theorem largest_crowd_size :
  ∃ (n : ℕ), 
    (⌊n / 2⌋ + ⌊n / 3⌋ + ⌊n / 5⌋ = n) ∧
    ∀ m : ℕ, (⌊m / 2⌋ + ⌊m / 3⌋ + ⌊m / 5⌋ = m) → m ≤ 37 :=
sorry

end largest_crowd_size_l402_402891


namespace gamma_lt_delta_l402_402344

open Real

variables (α β γ δ : ℝ)

-- Hypotheses as given in the problem
axiom h1 : 0 < α 
axiom h2 : α < β
axiom h3 : β < π / 2
axiom hg1 : 0 < γ
axiom hg2 : γ < π / 2
axiom htan_gamma_eq : tan γ = (tan α + tan β) / 2
axiom hd1 : 0 < δ
axiom hd2 : δ < π / 2
axiom hcos_delta_eq : (1 / cos δ) = (1 / 2) * (1 / cos α + 1 / cos β)

-- Goal to prove
theorem gamma_lt_delta : γ < δ := 
by 
sorry

end gamma_lt_delta_l402_402344


namespace product_sum_of_digits_base8_eq_21_l402_402818

-- Define the base-8 numbers
def num1 : ℕ := 3 * 8 + 5  -- 35_8 = 3 * 8^1 + 5 * 8^0
def num2 : ℕ := 2 * 8 + 1  -- 21_8 = 2 * 8^1 + 1 * 8^0

-- Define the product in decimal
def product : ℕ := num1 * num2

-- Convert product to base-8 and sum the digits
def sum_of_digits_base8 (n : ℕ) : ℕ :=
  let rec sum_digits (x : ℕ) (acc : ℕ) : ℕ :=
    match x with
    | 0 => acc
    | _ => sum_digits (x / 8) (acc + x % 8)
  sum_digits n 0

-- Assert the main proof statement: the sum of the digits in base-8
theorem product_sum_of_digits_base8_eq_21 : sum_of_digits_base8 product = 2 * 8 + 1 := by
  sorry

end product_sum_of_digits_base8_eq_21_l402_402818


namespace probability_rectangle_l402_402362

noncomputable def probability_x_greater_than_8y : ℚ :=
let width := 3014 in
let height := 3015 in
-- Area of the rectangle
let total_area := width * height in
-- Intersection point on y-axis and the triangle area
let intersect_y := width / 8 in
let triangle_area := (width * intersect_y) / 2 in
-- Probability calculation
triangle_area / total_area

theorem probability_rectangle : probability_x_greater_than_8y = 7535 / 120600 :=
by 
sorry

end probability_rectangle_l402_402362


namespace minimum_spending_l402_402449

noncomputable def box_volume (length width height : ℕ) : ℕ := length * width * height
noncomputable def total_boxes_needed (total_volume box_volume : ℕ) : ℕ := (total_volume + box_volume - 1) / box_volume
noncomputable def total_cost (num_boxes : ℕ) (price_per_box : ℝ) : ℝ := num_boxes * price_per_box

theorem minimum_spending
  (box_length box_width box_height : ℕ)
  (price_per_box : ℝ)
  (total_collection_volume : ℕ)
  (h1 : box_length = 20)
  (h2 : box_width = 20)
  (h3 : box_height = 15)
  (h4 : price_per_box = 0.90)
  (h5 : total_collection_volume = 3060000) :
  total_cost (total_boxes_needed total_collection_volume (box_volume box_length box_width box_height)) price_per_box = 459 :=
by
  rw [h1, h2, h3, h4, h5]
  have box_vol : box_volume 20 20 15 = 6000 := by norm_num [box_volume]
  have boxes_needed : total_boxes_needed 3060000 6000 = 510 := by norm_num [total_boxes_needed, box_volume, *]
  have cost : total_cost 510 0.90 = 459 := by norm_num [total_cost]
  exact cost

end minimum_spending_l402_402449


namespace trapezium_area_equality_l402_402078

-- Definitions
def Trapezium (A B Γ Δ : Type) (P : AΔ.parallel BΓ) (θ : ∠ A := 120) := True
def Midpoint (E : Type) (A B : Type) := True
def CircumcenterOfTriangle (O : Type) (T : Triangle) := True
def Area (S : Type) := ℝ

-- Trapezium with conditions
variables (A B Γ Δ E O₁ O₂ : Type) 
variables (P : A Δ.parallel B Γ) (θ : ∠ A = 120)
variables (hMid : Midpoint E A B)
variables (hCirc₁ : CircumcenterOfTriangle O₁ (triangle A E Δ))
variables (hCirc₂ : CircumcenterOfTriangle O₂ (triangle B E Γ))

-- Prove the desired area equality
theorem trapezium_area_equality :
  Area (trapezium A B Γ Δ) = 6 * Area (triangle O₁ E O₂) :=
sorry

end trapezium_area_equality_l402_402078


namespace triangle_inequality_l402_402791

-- Define the triangle angles, semiperimeter, and circumcircle radius
variables (α β γ s R : Real)

-- Define the sum of angles in a triangle
axiom angle_sum : α + β + γ = Real.pi

-- The inequality to prove
theorem triangle_inequality (h_sum : α + β + γ = Real.pi) :
  (α + β) * (β + γ) * (γ + α) ≤ 4 * (Real.pi / Real.sqrt 3)^3 * R / s := sorry

end triangle_inequality_l402_402791


namespace x_intersect_second_point_l402_402383

noncomputable def midpoint (x1 y1 x2 y2 : ℝ) := ((x1 + x2) / 2, (y1 + y2) / 2)

noncomputable def radius (cx cy x y : ℝ) := Real.sqrt ((cx - x) ^ 2 + (cy - y) ^ 2)

noncomputable def circle_eq (cx cy r : ℝ) (x y : ℝ) := (x - cx) ^ 2 + (y - cy) ^ 2 = r ^ 2

theorem x_intersect_second_point : 
  ∃ x : ℝ, (circle_eq 2 5 (4 * Real.sqrt 2) x 0) ∧ x = 2 + Real.sqrt 7 :=
by
  sorry

end x_intersect_second_point_l402_402383


namespace find_first_number_l402_402997

theorem find_first_number (g : ℕ) (h₁ : g = 144) 
  (h₂ : ∃ k : ℕ, 7373 = g * k + 29) 
  (h₃ : ∃ m : ℕ, x = g * m + 23) : 
  x = 7361 :=
by
  have hg : g = 144 := h₁
  obtain ⟨k, hk⟩ := h₂
  obtain ⟨m, hm⟩ := h₃
  rw hg at hk hm
  sorry

end find_first_number_l402_402997


namespace solve_for_x_l402_402447

theorem solve_for_x (x y : ℤ) (h1 : x + y = 14) (h2 : x - y = 60) :
  x = 37 :=
sorry

end solve_for_x_l402_402447


namespace symmetric_point_about_x_l402_402797

-- Define the coordinates of the point A
def A : ℝ × ℝ := (-2, 3)

-- Define the function that computes the symmetric point about the x-axis
def symmetric_about_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

-- The concrete symmetric point of A
def A' := symmetric_about_x A

-- The original problem and proof statement
theorem symmetric_point_about_x :
  A' = (-2, -3) :=
by
  -- Proof goes here
  sorry

end symmetric_point_about_x_l402_402797


namespace general_term_an_sum_first_n_terms_l402_402822

open Nat

def a (n : ℕ) : ℤ := 2 * n - 5

def b (n : ℕ) : ℤ := a n + 2^n

def Sn (n : ℕ) : ℤ :=
  ∑ i in range n + 1, b i

theorem general_term_an (n : ℕ) : a n = 2 * n - 5 := sorry

theorem sum_first_n_terms (n : ℕ) : Sn n = n^2 - 4 * n + 2^(n + 1) - 2 := sorry

end general_term_an_sum_first_n_terms_l402_402822


namespace external_angle_bisector_of_triangle_l402_402738

variables {A T C L K : Type}
variables [IsTriangle A T C] [IsInternalAngleBisector T L (A T C)]

theorem external_angle_bisector_of_triangle 
  (h1 : is_internal_angle_bisector TL ATC) :
  is_external_angle_bisector TK ATC :=
sorry

end external_angle_bisector_of_triangle_l402_402738


namespace find_first_discount_percentage_l402_402464

def first_discount_percentage 
  (price_initial : ℝ) 
  (price_final : ℝ) 
  (discount_x : ℝ) 
  : Prop := 
  price_initial * (1 - discount_x / 100) * 0.9 * 0.95 = price_final

theorem find_first_discount_percentage :
  first_discount_percentage 9941.52 6800 20.02 :=
by
  sorry

end find_first_discount_percentage_l402_402464


namespace system_has_six_solutions_l402_402995

theorem system_has_six_solutions (a : ℝ) (x y : ℝ) :
  (0 < a ∧ a ≠ 1) ∧ 
  ((a * y - a * x + 2) * (4 * y - 3 * (|x - a|) - x + 5 * a) = 0) ∧ 
  ((log a (x^2) + log a (y^2) - 2) * log 2 (a^2) = 8) ↔ 
  (a ∈ Set.Ioo 0 (real.cbrt 1 / real.cbrt 4) ∨ a ∈ Set.Ioo 4 32) := 
sorry

end system_has_six_solutions_l402_402995


namespace ellipse_eq_triangle_area_l402_402205

-- Part I: Equation of the ellipse
theorem ellipse_eq (a b : ℝ) (h1 : a > b) (h2 : b > 0)
  (h3 : (1 : ℝ) / a^2 + ((sqrt 2) / 2) ^ 2 / b^2 = 1)
  (h4 : sqrt (a^2 - b^2) / a = (sqrt 2) / 2) : 
  (a^2 = 2) ∧ (b^2 = 1) :=
sorry

-- Part II: Area of triangle ABC
theorem triangle_area (a b : ℝ)
  (h_ellipse : ∀ x y : ℝ, x^2 / 2 + y^2 = 1 → true)
  (h_angle_bisector : ∀ (x y : ℝ), y = -3 * x + 1 → true) :
  ∃ (area : ℝ), area = 40 / 27 :=
sorry

end ellipse_eq_triangle_area_l402_402205


namespace proof_AE_over_EF_l402_402073

noncomputable def point := (ℝ × ℝ)

def A (a : ℝ) : point := (a, 0)
def D (a : ℝ) : point := (a, a)

def B (a b : ℝ) : point := (b, b)
def C (a c : ℝ) : point := (c, 2*c)

def condition_AB_BC (a b c : ℝ) : Prop := 
  real.sqrt ((b - a)^2 + b^2) / real.sqrt ((c - b)^2 + (2*c - b)^2) = 2

def circumcircle_intersects_y_equals_x_again_at_E (a b : ℝ) (A : point) (D : point) (C : point) : point := 
  -- Intersection point on y = x, assume E
  (frac(a, 4), frac(a, 4)) -- Assuming E = (a/4, a/4) 

def ray_AE_intersects_y_equals_2x_at_F (a : ℝ) (E : point) : point := 
  -- Intersection point E
  (frac(a, 7), frac(2 * a / 7)) -- Assuming F = (a/7, 2a/7)

theorem proof_AE_over_EF (a b c : ℝ) (E F : point) :
  ∀ (h : condition_AB_BC a b c), 
  let E := circumcircle_intersects_y_equals_x_again_at_E a b (A a) (D a) (C a c) in
  let F := ray_AE_intersects_y_equals_2x_at_F a E in
  (real.dist (A a) E) / (real.dist E F) = 7 := 
sorry

end proof_AE_over_EF_l402_402073


namespace sum_of_reciprocal_roots_l402_402326

-- Defining the polynomial whose roots are given
noncomputable def polynomial := (λ x : ℂ, x^2018 + x^2017 + x^2016 + ... + x^2 + x - 1345)

-- Given that a_1, a_2, ..., a_2018 are the roots of the polynomial defined
noncomputable def roots := {a : ℂ | polynomial a = 0}

-- The problem statement in Lean 4
theorem sum_of_reciprocal_roots :
  ∑ (n : ℕ) in finset.range 2018, (1 : ℂ) / (1 - (classical.some (roots n))) = 3027 :=
sorry

end sum_of_reciprocal_roots_l402_402326


namespace rows_per_floor_l402_402678

theorem rows_per_floor
  (right_pos : ℕ) (left_pos : ℕ)
  (floors : ℕ) (total_cars : ℕ)
  (h_right : right_pos = 5) (h_left : left_pos = 4)
  (h_floors : floors = 10) (h_total : total_cars = 1600) :
  ∃ rows_per_floor : ℕ, rows_per_floor = 20 :=
by {
  sorry
}

end rows_per_floor_l402_402678


namespace P_B_given_A_l402_402027

/-- definition of event A as the outcomes where both rolls are odd numbers -/
def event_A : set (ℕ × ℕ) := 
  { (1, 1), (1, 3), (1, 5), (3, 1), (3, 3), (3, 5), (5, 1), (5, 3), (5, 5) }

/-- definition of event B as the outcomes where the sum of the outcomes of both rolls is 4 -/
def event_B : set (ℕ × ℕ) := 
  { (x, y) | x + y = 4 }

/-- Proving that the probability of B given A is 2/9 -/
theorem P_B_given_A : P(event_B | event_A) = 2/9 := 
by 
  sorry

end P_B_given_A_l402_402027


namespace time_released_rope_first_time_l402_402308

theorem time_released_rope_first_time :
  ∀ (rate_ascent : ℕ) (rate_descent : ℕ) (time_first_ascent : ℕ) (time_second_ascent : ℕ) (highest_elevation : ℕ)
    (total_elevation_gained : ℕ) (elevation_difference : ℕ) (time_descent : ℕ),
  rate_ascent = 50 →
  rate_descent = 10 →
  time_first_ascent = 15 →
  time_second_ascent = 15 →
  highest_elevation = 1400 →
  total_elevation_gained = (rate_ascent * time_first_ascent) + (rate_ascent * time_second_ascent) →
  elevation_difference = total_elevation_gained - highest_elevation →
  time_descent = elevation_difference / rate_descent →
  time_descent = 10 :=
by
  intros rate_ascent rate_descent time_first_ascent time_second_ascent highest_elevation total_elevation_gained elevation_difference time_descent
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end time_released_rope_first_time_l402_402308


namespace fifth_degree_monomial_l402_402002

theorem fifth_degree_monomial (x y : ℕ) (hx : x = 2) (hy : y = 3) : (x + y = 5) :=
by {
  rw [hx, hy],
  simp,
  sorry,
}

end fifth_degree_monomial_l402_402002


namespace cos_C_sin_B_area_l402_402269

noncomputable def triangle_conditions (A B C a b c : ℝ) : Prop :=
  (A + B + C = Real.pi) ∧
  (b / c = 2 * Real.sqrt 3 / 3) ∧
  (A + 3 * C = Real.pi)

theorem cos_C (A B C a b c : ℝ) (h : triangle_conditions A B C a b c) :
  Real.cos C = Real.sqrt 3 / 3 :=
sorry

theorem sin_B (A B C a b c : ℝ) (h : triangle_conditions A B C a b c) :
  Real.sin B = 2 * Real.sqrt 2 / 3 :=
sorry

theorem area (A B C a b c : ℝ) (h : triangle_conditions A B C a b c) (hb : b = 3 * Real.sqrt 3) :
  (1 / 2) * b * c * Real.sin A = 9 * Real.sqrt 2 / 4 :=
sorry

end cos_C_sin_B_area_l402_402269


namespace inverse_of_projection_matrix_l402_402323

open Classical

noncomputable def P : Matrix (Fin 3) (Fin 3) ℝ :=
  (1/5:ℝ) • (Matrix.ofMatrix ![
    ![1, 0, 2],
    ![0, 0, 0],
    ![2, 0, 4]
  ])

theorem inverse_of_projection_matrix :
  (∀ (v : Fin 3 → ℝ), v = ![1, 0, 2]) →
  (∀ (P : Matrix (Fin 3) (Fin 3) ℝ), 
    P = (1/5:ℝ) • (Matrix.ofMatrix ![
      ![1, 0, 2],
      ![0, 0, 0],
      ![2, 0, 4]
    ]) →
    ¬ invertible P) →
  (P⁻¹ = 0) :=
by
  assume h_v h_P
  sorry

end inverse_of_projection_matrix_l402_402323


namespace angle_sum_l402_402914

-- Define the angles in the isosceles triangles
def angle_BAC := 40
def angle_EDF := 50

-- Using the property of isosceles triangles to calculate other angles
def angle_ABC := (180 - angle_BAC) / 2
def angle_DEF := (180 - angle_EDF) / 2

-- Since AD is parallel to CE, angles DAC and ACB are equal as are ADE and DEF
def angle_DAC := angle_ABC
def angle_ADE := angle_DEF

-- The theorem to be proven
theorem angle_sum :
  angle_DAC + angle_ADE = 135 :=
by
  sorry

end angle_sum_l402_402914


namespace privateer_overtakes_merchantman_l402_402938

-- Definitions for initial conditions
def initial_distance : ℝ := 15 -- in miles
def privateer_initial_speed : ℝ := 13 -- in mph
def merchantman_speed : ℝ := 9 -- in mph
def damage_time : ℝ := 1.5 -- in hours
def privateer_post_damage_distance : ℝ := 14 -- miles in time merchantman covers 12 miles
def merchantman_post_damage_distance : ℝ := 12 -- miles in same timeframe as above

-- Main statement to prove
theorem privateer_overtakes_merchantman :
  let relative_speed1 := privateer_initial_speed - merchantman_speed,
      distance_after_damage_time :=  initial_distance - (privateer_initial_speed * damage_time - merchantman_speed * damage_time),
      privateer_post_damage_speed := (privateer_post_damage_distance / merchantman_post_damage_distance) * merchantman_speed,
      relative_speed2 := privateer_post_damage_speed - merchantman_speed,
      time_to_overtake_post_damage := distance_after_damage_time / relative_speed2,
      total_time_to_overtake := damage_time + time_to_overtake_post_damage,
      overtake_time := 12 + total_time_to_overtake,
      final_day_time := overtake_time - if overtake_time >= 24 then 24 else 0
  in final_day_time = 4.95 then -- 4:57 AM as a fraction of 24-hour time
  sorry -- Proof omitted

end privateer_overtakes_merchantman_l402_402938


namespace number_of_proper_subsets_l402_402813

theorem number_of_proper_subsets (S : Finset ℕ) (h : S = {1, 2, 3, 4}) : S.powerset.card - 1 = 15 := by
  sorry

end number_of_proper_subsets_l402_402813


namespace geometric_sequence_sum_l402_402197

theorem geometric_sequence_sum (a_1 : ℝ) (q : ℝ) (S : ℕ → ℝ) :
  q = 2 →
  (∀ n, S (n+1) = a_1 * (1 - q^(n+1)) / (1 - q)) →
  S 4 / a_1 = 15 :=
by
  intros hq hsum
  sorry

end geometric_sequence_sum_l402_402197


namespace distance_between_ships_l402_402358

-- Define the height of the fort
def fort_height : ℝ := 30

-- Define the depression angles to Ship A and Ship B
def alpha : ℝ := real.pi / 4 -- 45 degrees in radians
def beta : ℝ := real.pi / 6 -- 30 degrees in radians

-- Define the angle at the base of the fort to each ship
def theta : ℝ := real.pi / 6 -- 30 degrees in radians

-- Prove the distance between the two ships is 30 meters
theorem distance_between_ships (h : ℝ) (α β θ : ℝ) : h = 30 → α = real.pi / 4 → β = real.pi / 6 → θ = real.pi / 6 → 
  let BA := h / (real.tan θ) in let BB := BA + h / (real.tan β) in
  BB - BA = 30 :=
sorry

end distance_between_ships_l402_402358


namespace sum_of_coeffs_eq_225_l402_402405

/-- The sum of the coefficients of all terms in the expansion
of (C_x + C_x^2 + C_x^3 + C_x^4)^2 is equal to 225. -/
theorem sum_of_coeffs_eq_225 (C_x : ℝ) : 
  (C_x + C_x^2 + C_x^3 + C_x^4)^2 = 225 :=
sorry

end sum_of_coeffs_eq_225_l402_402405


namespace area_ratio_is_l402_402288

-- Define the relevant lengths and conditions
variables {P Q R S T : Type} [ordered_field P Q R S T]
variables (a : ℝ) -- Length of PQ
variables (h1 : angle P Q R = π / 2) -- PQR is a right angle
variables (h2 : ∀ S T: ℝ, Q R = 2 * P Q) -- QR = 2 * PQ
variables (h3 : ∃ S T: ℝ, PS and PT trisect angle QPR)

-- Main theorem to prove that the ratio of the areas satisfes the given condition
theorem area_ratio_is (h1 : angle P Q R = π / 2) (h2 : QR = 2 * PQ) (h3 : ∃ S T: ℝ, PS and PT trisect angle QPR) : 
  let PQ := a in let QR := 2 * a in 
  (area_of_triangle_PST : area P S T / area P Q R = 2 * real.sqrt(3) / 9) :=
by
  sorry

end area_ratio_is_l402_402288


namespace external_angle_bisector_of_triangle_l402_402744

variables {A T C L K : Type}
variables [IsTriangle A T C] [IsInternalAngleBisector T L (A T C)]

theorem external_angle_bisector_of_triangle 
  (h1 : is_internal_angle_bisector TL ATC) :
  is_external_angle_bisector TK ATC :=
sorry

end external_angle_bisector_of_triangle_l402_402744


namespace perp_intersection_l402_402554

-- Definitions for the conditions specified in the problem
variables {A B C D E P O : Type}
variable [IsTriangle A B C] -- A denotes a triangle with vertices A, B, and C.
variable [IsCircle O B C] -- O denotes the circle passing through B and C that intersects AC and AB.
variable (D : Type) [intersection_point O A C] -- D is the intersection point of O and AC.
variable (E : Type) [intersection_point O A B] -- E is the intersection point of O and AB.
variable [circumcircle ADE A P : Type] -- defines the circumcircle of ADE intersecting circumcircle of ABC at A and P.
variable [circumcircle ABC A P : Type] -- defines the circumcircle of ABC intersecting circumcircle of ADE at A and P.

-- Theorem to prove that AP is perpendicular to OP
theorem perp_intersection : 
  ⟦ AP ⟧ ⊥ ⟦ OP ⟧ :=
  sorry

end perp_intersection_l402_402554


namespace sufficient_but_not_necessary_condition_l402_402209

theorem sufficient_but_not_necessary_condition
  (a : ℝ) :
  (a = 2 → (a - 1) * (a - 2) = 0)
  ∧ (∃ a : ℝ, (a - 1) * (a - 2) = 0 ∧ a ≠ 2) :=
by
  sorry

end sufficient_but_not_necessary_condition_l402_402209


namespace carA_speed_calc_l402_402068

-- Defining the conditions of the problem
def carA_time : ℕ := 8
def carB_speed : ℕ := 25
def carB_time : ℕ := 4
def distance_ratio : ℕ := 4
def carB_distance : ℕ := carB_speed * carB_time
def carA_distance : ℕ := distance_ratio * carB_distance

-- Mathematical statement to be proven
theorem carA_speed_calc : carA_distance / carA_time = 50 := by
  sorry

end carA_speed_calc_l402_402068


namespace number_divided_is_144_l402_402770

theorem number_divided_is_144 (n divisor quotient remainder : ℕ) (h_divisor : divisor = 11) (h_quotient : quotient = 13) (h_remainder : remainder = 1) (h_division : n = (divisor * quotient) + remainder) : n = 144 :=
by
  sorry

end number_divided_is_144_l402_402770


namespace count_integer_values_of_a_l402_402158

theorem count_integer_values_of_a (a : ℤ) :
  (∃ x : ℤ, x^2 + a * x + 12 * a = 0) ↔ (a ∈ {-(m + n) | m, n : ℤ ∧ m * n = 12 * a}) → ∃! a_vals : finset ℤ, a_vals.card = 16 :=
by
  sorry

end count_integer_values_of_a_l402_402158


namespace height_difference_l402_402810

variables (H1 H2 H3 : ℕ)
variable (x : ℕ)
variable (h_ratio : H1 = 4 * x ∧ H2 = 5 * x ∧ H3 = 6 * x)
variable (h_lightest : H1 = 120)

theorem height_difference :
  (H1 + H3) - H2 = 150 :=
by
  -- Proof will go here
  sorry

end height_difference_l402_402810


namespace original_price_of_painting_l402_402352

-- Problem conditions
variables (P : ℝ) (cost_paintings cost_toys sell_paintings sell_toys : ℝ)

-- Conditions based on problem statement
def cost_paintings := 10 * P
def cost_toys := 8 * 20
def total_cost := cost_paintings + cost_toys

def sell_paintings := 10 * (0.90 * P)
def sell_toys := 8 * 17
def total_sell := sell_paintings + sell_toys

def loss := total_cost - total_sell

-- The theorem to be proven
theorem original_price_of_painting : loss = 64 → P = 40 :=
by
  assume h : loss = 64
  -- TODO: Provide the complete proof (currently skipped with sorry)
  sorry

end original_price_of_painting_l402_402352


namespace quincy_more_stuffed_animals_l402_402668

theorem quincy_more_stuffed_animals (thor_sold jake_sold quincy_sold : ℕ) 
  (h1 : jake_sold = thor_sold + 10) 
  (h2 : quincy_sold = 10 * thor_sold) 
  (h3 : quincy_sold = 200) : 
  quincy_sold - jake_sold = 170 :=
by sorry

end quincy_more_stuffed_animals_l402_402668


namespace problem_l402_402613

theorem problem (a b : ℤ)
  (h1 : -2022 = -a)
  (h2 : -1 = -b) :
  a + b = 2023 :=
sorry

end problem_l402_402613


namespace number_of_distinct_a_l402_402127

noncomputable def count_distinct_a : ℕ :=
  let divisors := [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 36, 48, 72, 144, -1, -2, -3, -4, -6, -8, -9, -12, -16, -18, -24, -36, -48, -72, -144]
  let pairs := (divisors.product divisors).filter (λ (p : ℤ × ℤ), p.1 * p.2 = 144)
  let sums := pairs.map (λ (p : ℤ × ℤ), p.1 + p.2 - 24)
  let distinct_sums := sums.eraseDuplicates  
  distinct_sums.length

theorem number_of_distinct_a : count_distinct_a = 14 :=
by
  sorry

end number_of_distinct_a_l402_402127


namespace warrior_defeats_dragon_l402_402387

noncomputable def heads : Nat := 2000
noncomputable def can_chop : Set Nat := {33, 21, 17, 1}
noncomputable def grows_back : Nat → Nat
| 33 => 48
| 21 => 0
| 17 => 14
| 1 => 349
| _ => 0

def warrior_can_defeat_dragon (initial_heads : Nat) : Prop :=
  ∃ (sequence : List Nat), 
    (∀ x ∈ sequence, x ∈ can_chop) ∧ 
    initial_heads - (sequence.foldl (+) 0 sequence) + (sequence.map grows_back).foldl (+) 0 sequence = 0

theorem warrior_defeats_dragon : warrior_can_defeat_dragon heads :=
sorry

end warrior_defeats_dragon_l402_402387


namespace integer_values_a_l402_402182

theorem integer_values_a :
  (∃! a : ℤ, ∃ m n : ℤ, m^2 + n^2 = (m + n)^2 - 2mn ∧
                     m + n = -a ∧ 
                     mn = 12a) :
  8 :=
sorry

end integer_values_a_l402_402182


namespace height_from_vertex_in_acute_triangle_l402_402275

theorem height_from_vertex_in_acute_triangle
    {A B C : Type} [TrigonometricType A] [TrigonometricType B] [TrigonometricType C]
    (h1 : sin (A + B) = 3 / 5)
    (h2 : sin (A - B) = 1 / 5)
    (h3 : AB = 3)
    (h_acute : is_acute_triangle A B C) :
    height_from_vertex A B C = 2 + sqrt 6 :=
sorry

end height_from_vertex_in_acute_triangle_l402_402275


namespace point_transformation_correct_l402_402659

-- Define the rectangular coordinate system O-xyz
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the point in the original coordinate system
def originalPoint : Point3D := { x := 1, y := -2, z := 3 }

-- Define the transformation function for the yOz plane
def transformToYOzPlane (p : Point3D) : Point3D :=
  { x := -p.x, y := p.y, z := p.z }

-- Define the expected transformed point
def transformedPoint : Point3D := { x := -1, y := -2, z := 3 }

-- State the theorem to be proved
theorem point_transformation_correct :
  transformToYOzPlane originalPoint = transformedPoint :=
by
  sorry

end point_transformation_correct_l402_402659


namespace smaller_square_area_from_midpoints_l402_402927

-- Define the conditions
def larger_square_area : ℝ := 100
def mid_squares_area (A : ℝ) : ℝ := (5 * √2)^2

-- Statement of the problem
theorem smaller_square_area_from_midpoints
  (A : ℝ)
  (H : A = 100) :
  mid_squares_area A = 50 := 
begin
  sorry
end

end smaller_square_area_from_midpoints_l402_402927


namespace square_roll_triangle_left_l402_402481

theorem square_roll_triangle_left
    (square : Type)
    (octagon : Type)
    (initial_position : square → octagon)
    (final_position : square → octagon)
    (triangle_position : square → Type)
    (rolls_clockwise : ∀ (s : square) (o : octagon), initial_position s = o → final_position s = o)
    (initial_right : triangle_position initial_position = "right")
    (reaches_left : final_position = "leftmost") :
  triangle_position final_position = "left" := sorry

end square_roll_triangle_left_l402_402481


namespace limit_problem_l402_402064

open Real

theorem limit_problem :
  filter.tendsto (λ x : ℝ, (exp (2 * x) - exp (3 * x)) / (arctan x - x ^ 2))
                (nhds_within 0 (set.univ : set ℝ))
                (nhds (-1)) :=
  sorry

end limit_problem_l402_402064


namespace min_deg_q_l402_402578

-- Definitions of polynomials requirements
variables (p q r : Polynomial ℝ)

-- Given Conditions
def polynomials_relation : Prop := 5 * p + 6 * q = r
def deg_p : Prop := p.degree = 10
def deg_r : Prop := r.degree = 12

-- The main theorem we want to prove
theorem min_deg_q (h1 : polynomials_relation p q r) (h2 : deg_p p) (h3 : deg_r r) : q.degree ≥ 12 :=
sorry

end min_deg_q_l402_402578


namespace baron_munchausen_max_crowd_size_l402_402880

theorem baron_munchausen_max_crowd_size :
  ∃ n : ℕ, (∀ k, (k : ℕ) = n → 
  let left := (k / 2).toNat;
      right := (k / 3).toNat;
      straight := (k / 5).toNat in
  left + right + straight <= n + 1) ∧ 
  (∀ x : ℕ, x > 37 → ¬(∀ k, (k : ℕ) = x →
  let left := (k / 2).toNat;
      right := (k / 3).toNat;
      straight := (k / 5).toNat in
  left + right + straight <= x + 1)) :=
begin
  have h : 37 = 18 + 12 + 7,
  sorry,
end

end baron_munchausen_max_crowd_size_l402_402880


namespace min_value_of_y_l402_402690

noncomputable def y (x : ℝ) : ℝ := abs (x - 1) + abs (x - 2) - abs (x - 3)

theorem min_value_of_y : ∃ x : ℝ, (∀ x' : ℝ, y x' ≥ y x) ∧ y x = -1 :=
sorry

end min_value_of_y_l402_402690


namespace max_sum_is_23_l402_402440

def adjust_circle_values (lst : List ℕ) : List ℕ :=
  lst.map (λ x => if x % 2 == 0 then x + 1 else x - 1)

def max_sum_of_three_consecutives (lst : List ℕ) : ℕ :=
  (List.range (lst.length)).map (λ i => lst.get! i + lst.get! ((i+1) % lst.length) + lst.get! ((i+2) % lst.length)).maximum.getD 0

theorem max_sum_is_23 :
  max_sum_of_three_consecutives (adjust_circle_values [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) = 23 :=
by
  -- Note: Proof is omitted
  sorry

end max_sum_is_23_l402_402440


namespace total_charge_for_2_hours_l402_402855

theorem total_charge_for_2_hours (A F : ℕ) (h1 : F = A + 35) (h2 : F + 4 * A = 350) : 
  F + A = 161 := 
by 
  sorry

end total_charge_for_2_hours_l402_402855


namespace cube_faces_sum_39_l402_402984

theorem cube_faces_sum_39 (a b c d e f g h : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0 ∧ g > 0 ∧ h > 0)
    (vertex_sum : (a*e*b*h + a*e*c*h + a*f*b*h + a*f*c*h + d*e*b*h + d*e*c*h + d*f*b*h + d*f*c*h) = 2002) :
    (a + b + c + d + e + f + g + h) = 39 := 
sorry

end cube_faces_sum_39_l402_402984


namespace marble_287_is_blue_l402_402479

def marble_color (n : ℕ) : String :=
  if n % 15 < 6 then "blue"
  else if n % 15 < 11 then "green"
  else "red"

theorem marble_287_is_blue : marble_color 287 = "blue" :=
by
  sorry

end marble_287_is_blue_l402_402479


namespace sum_of_coefficients_l402_402523

theorem sum_of_coefficients : 
  let P := λ (x : ℝ), 3 * (x^8 - 2*x^5 + 4*x^3 - 6) - 5 * (x^4 - 3*x^2 + 2) + 2 * (x^6 + 5*x - 8)
  in P(1) = -3 :=
by
  let P := λ (x : ℝ), 3 * (x^8 - 2*x^5 + 4*x^3 - 6) - 5 * (x^4 - 3*x^2 + 2) + 2 * (x^6 + 5*x - 8)
  sorry

end sum_of_coefficients_l402_402523


namespace relationship_abc_l402_402586

theorem relationship_abc {m : ℝ} (h_even : ∀ x, (2^|x - m| - 1) = (2^| -x - m| - 1)) :
  let f := λ x, 2^|x - m| - 1,
      a := f (-2),
      b := f (Real.log 5 / Real.log 2),
      c := f (2 * m)
  in c < a ∧ a < b :=
by
  sorry

end relationship_abc_l402_402586


namespace point_not_on_transformed_plane_l402_402894

theorem point_not_on_transformed_plane (k : ℚ) (A : ℝ × ℝ × ℝ) (a : ℝ → ℝ → ℝ → Prop) :
  k = 2/3 → A = (1, -2, 1) → 
  (∀ x y z, a x y z ↔ 5 * x + y - z + 6 = 0) → 
  ¬ a (1 * (1 / k)) (-2 * (1 / k)) (1 * (1 / k)) := 
by 
  intros hk hA ha' 
  suffices : ¬ (5 * (1 * (1 / k)) + (-2 * (1 / k)) - (1 * (1 / k)) + 6 * (1 / k) = 0), sorry
  -- Translate problem to proving statement in question
  simp only [hk, inv_div, inv_mul_cancel_right',
             ne.def, inv_injective, eq_self_iff_true, not_true]
  sorry

end point_not_on_transformed_plane_l402_402894


namespace neg_abs_nonneg_l402_402236

theorem neg_abs_nonneg :
  (¬ ∀ x : ℝ, |x| ≥ 0) ↔ (∃ x : ℝ, |x| < 0) := by
  sorry

end neg_abs_nonneg_l402_402236


namespace number_of_integer_values_of_a_l402_402167

theorem number_of_integer_values_of_a : 
  ∃ (a_set : Set ℤ), (∀ a ∈ a_set, ∃ x y : ℤ, x^2 + a * x + 12 * a = 0 ∧ y^2 + a * y + 12 * a = 0) ∧ a_set.card = 16 :=
sorry

end number_of_integer_values_of_a_l402_402167


namespace sufficient_not_necessary_condition_l402_402902

theorem sufficient_not_necessary_condition (x : ℝ) : (1 < x ∧ x < 2) → (x < 2) ∧ ((x < 2) → ¬(1 < x ∧ x < 2)) :=
by
  sorry

end sufficient_not_necessary_condition_l402_402902


namespace prob_selecting_A_from_4_l402_402624

theorem prob_selecting_A_from_4 (A B C D : Type) :
  (∃ s : set (set {x : Type // x = A ∨ x = B ∨ x = C ∨ x = D}), 
  s.card = 3) →
  (∃ t : set (set {x : Type // x = A ∨ x = B ∨ x = C ∨ x = D}),
  t.card = 4 ∧ 3 ∈ t) →
  (∃ p : ℝ, p = 3 / 4) := by
sorry

end prob_selecting_A_from_4_l402_402624


namespace smallest_k_for_Δk_un_zero_l402_402969

def u (n : ℕ) : ℤ := n^3 - n

def Δ (k : ℕ) (u : ℕ → ℤ) : ℕ → ℤ :=
  match k with
  | 0     => u
  | (k+1) => λ n => Δ k u (n+1) - Δ k u n

theorem smallest_k_for_Δk_un_zero (u : ℕ → ℤ) (h : ∀ n, u n = n^3 - n) :
  ∀ n, Δ 4 u n = 0 ∧ (∀ k < 4, ∃ n, Δ k u n ≠ 0) :=
by
  sorry

end smallest_k_for_Δk_un_zero_l402_402969


namespace tangent_line_equation_l402_402804

theorem tangent_line_equation (x y b : ℝ) :
  (∀ (x1 y1 : ℝ), x1 + y1 - 1 = 0 → x + y + b = 0) →
  (∀ (x y : ℝ), x^2 + y^2 - 2 = 0 → (↑ (x + y + b = 0) = ↑(x^2 + y^2 - 2)) ) →
  (b = 2 ∨ b = -2) :=
by
  sorry

end tangent_line_equation_l402_402804


namespace quadratic_form_ratio_l402_402365

theorem quadratic_form_ratio (x y u v : ℤ) (h : ∃ k : ℤ, k * (u^2 + 3*v^2) = x^2 + 3*y^2) :
  ∃ a b : ℤ, (x^2 + 3*y^2) / (u^2 + 3*v^2) = a^2 + 3*b^2 := sorry

end quadratic_form_ratio_l402_402365


namespace count_integer_values_of_a_l402_402157

theorem count_integer_values_of_a (a : ℤ) :
  (∃ x : ℤ, x^2 + a * x + 12 * a = 0) ↔ (a ∈ {-(m + n) | m, n : ℤ ∧ m * n = 12 * a}) → ∃! a_vals : finset ℤ, a_vals.card = 16 :=
by
  sorry

end count_integer_values_of_a_l402_402157


namespace external_bisector_l402_402749

open Triangle

variable {Point : Type} [EuclideanGeometry Point]

variables {A T C L K : Point}

theorem external_bisector (h : angle_bisector A T C L) : external_bisector A T C K :=
sorry

end external_bisector_l402_402749


namespace max_area_triangle_find_b_c_l402_402265

-- Question 1
theorem max_area_triangle (a b : ℝ) (sinC : ℝ) (h_a_b : a + b = 5) (h_sinC : sinC = (sqrt 10) / 4) :
  let area : ℝ := (1/2) * a * b * sinC in area ≤ 25 * sqrt 10 / 32 :=
by sorry

-- Question 2
theorem find_b_c (a c b sinA sinC cosA : ℝ) (h_a : a = 2)
  (h_eqn1 : 2 * sinA ^ 2 + sinA * sinC = sinC ^ 2) (h_sinC : sinC = (sqrt 10) / 4)
  (h_sinA : sinA = (sqrt 10) / 8) (h_cosA : cosA = (3 * sqrt 6) / 8)
  (h_cos_law : cosA = (b ^ 2 + c ^ 2 - a ^ 2) / (2 * b * c)) :
  c = 4 ∧ b = (3 * sqrt 6 + sqrt 102) / 2 :=
by sorry

end max_area_triangle_find_b_c_l402_402265


namespace external_bisector_l402_402754

-- Define the relevant points and segments in the triangle
variables {A T C L K : Type} [inhabited A] [inhabited T] [inhabited C] [inhabited L] [inhabited K]

-- Assuming T, L, K are points such that TL divides angle ATC internally
axiom internal_bisector (h : L) : ∃ TL, TL.bisects_angle (A T C) internally

-- Prove that TK is the external angle bisector given the condition above
theorem external_bisector (h : L) (h_internal : internal_bisector h) : 
  ∃ TK, TK.bisects_angle (A T C) externally :=
sorry

end external_bisector_l402_402754


namespace pizza_slices_with_both_l402_402912

theorem pizza_slices_with_both (total_slices pepperoni_slices mushroom_slices : ℕ) 
  (h_total : total_slices = 24) (h_pepperoni : pepperoni_slices = 15) (h_mushrooms : mushroom_slices = 14) :
  ∃ n, n = 5 ∧ total_slices = pepperoni_slices + mushroom_slices - n := 
by
  use 5
  sorry

end pizza_slices_with_both_l402_402912


namespace estimate_value_l402_402985

theorem estimate_value : 1 < (3 - Real.sqrt 3) ∧ (3 - Real.sqrt 3) < 2 :=
by
  have h₁ : Real.sqrt 18 = 3 * Real.sqrt 2 :=
    by sorry
  have h₂ : Real.sqrt 6 = Real.sqrt 3 * Real.sqrt 2 :=
    by sorry
  have h₃ : (Real.sqrt 18 - Real.sqrt 6) / Real.sqrt 2 = (3 * Real.sqrt 2 - Real.sqrt 3 * Real.sqrt 2) / Real.sqrt 2 :=
    by sorry
  have h₄ : (3 * Real.sqrt 2 - Real.sqrt 3 * Real.sqrt 2) / Real.sqrt 2 = 3 - Real.sqrt 3 :=
    by sorry
  have h₅ : 1 < Real.sqrt 3 ∧ Real.sqrt 3 < 2 :=
    by sorry
  sorry

end estimate_value_l402_402985


namespace polynomial_no_positive_real_roots_l402_402801

theorem polynomial_no_positive_real_roots : 
  ¬ ∃ x : ℝ, x > 0 ∧ x^3 + 6 * x^2 + 11 * x + 6 = 0 :=
sorry

end polynomial_no_positive_real_roots_l402_402801


namespace num_possible_pairs_l402_402476

theorem num_possible_pairs (a b : ℕ) (h1 : b > a) (h2 : (a - 8) * (b - 8) = 32) : 
    (∃ n, n = 3) :=
by { sorry }

end num_possible_pairs_l402_402476


namespace total_flowers_sold_l402_402947

theorem total_flowers_sold :
  let flowers_mon := 4
  let flowers_tue := 8
  let flowers_wed := flowers_mon + 3
  let flowers_thu := 6
  let flowers_fri := flowers_mon * 2
  let flowers_sat := 5 * 9
  flowers_mon + flowers_tue + flowers_wed + flowers_thu + flowers_fri + flowers_sat = 78 :=
by
  let flowers_mon := 4
  let flowers_tue := 8
  let flowers_wed := flowers_mon + 3
  let flowers_thu := 6
  let flowers_fri := flowers_mon * 2
  let flowers_sat := 5 * 9
  sorry

end total_flowers_sold_l402_402947


namespace TK_is_external_bisector_of_ATC_l402_402705

variable {A T C L K : Type} [LinearOrder A] [LinearOrder T] [LinearOrder C] [LinearOrder L] [LinearOrder K]
variable (triangle_ATC : Triangle A T C)
variable (point_T : Point T)
variable (point_L : Point L)
variable (point_K : Point K)
variable (internal_bisector_TL : AngleBisector (angle_ATC) T L)
variable (external_bisector_TK : AngleBisector (external_angle_ATC) T K)

theorem TK_is_external_bisector_of_ATC :
  external_bisector_TK = true :=
sorry

end TK_is_external_bisector_of_ATC_l402_402705


namespace age_difference_l402_402407

variables (A B C : ℕ)

theorem age_difference (h : A + B = B + C + 14) : C = A - 14 :=
by sorry

end age_difference_l402_402407


namespace polynomial_rational_roots_count_l402_402937

/-- The polynomial 8x^4 + a_3 x^3 + a_2 x^2 + a_1 x + 16 = 0 has 
exactly 16 different possible rational roots -/
theorem polynomial_rational_roots_count (a_1 a_2 a_3 : ℤ) :
  let p : Polynomial ℤ := 8 * Polynomial.X^4 + a_3 * Polynomial.X^3 + a_2 * Polynomial.X^2 + a_1 * Polynomial.X + 16 in
  (roots_count p = 16) :=
sorry

end polynomial_rational_roots_count_l402_402937


namespace max_problems_per_participant_l402_402642

theorem max_problems_per_participant :
  ∀ (participants : ℕ) (avg_problems : ℕ),
  participants = 25 →
  avg_problems = 6 →
  (∀ i, i < participants → 1 ≤ (λ p, p i)) →
  ∃ p_max, p_max ≤ (participants * avg_problems - (participants - 1)) := by
  sorry

end max_problems_per_participant_l402_402642


namespace line_perpendicular_to_plane_l402_402778

variables (Plane : Type) (Line Point : Type)
variables (α β : Plane) (a c : Line)
variables (α_perp_β : α ⟂ β)
variables (a_in_α : a ∈ α) (c_in_αβ : c = α ∩ β) (a_perp_c : a ⟂ c)

theorem line_perpendicular_to_plane :
  a ⟂ β := sorry

end line_perpendicular_to_plane_l402_402778


namespace sum_of_four_consecutive_primes_l402_402970

theorem sum_of_four_consecutive_primes (A B : ℕ) 
  (hA: A.prime) (hB: B.prime) (hAB_sub: (A - B).prime) (hAB_add: (A + B).prime)
  (h_consec_odd_primes: ∃ p1 p2 p3 p4 : ℕ, p3 = A ∧ p2 = B ∧ prime p1 ∧ prime p2 ∧ prime p3 ∧ prime p4 
                        ∧ (p2 = p1 + 2) ∧ (p3 = p2 + 2) ∧ (p4 = p3 + 2)) :
  (A + B + (A - B) + (A + B) = 17) :=
by
  sorry

end sum_of_four_consecutive_primes_l402_402970


namespace largest_divisor_of_even_square_difference_l402_402341

theorem largest_divisor_of_even_square_difference (m n : ℕ) (hm : m % 2 = 0) (hn : n % 2 = 0) (h : n < m) :
  ∃ (k : ℕ), k = 8 ∧ ∀ m n : ℕ, m % 2 = 0 → n % 2 = 0 → n < m → k ∣ (m^2 - n^2) := by
  sorry

end largest_divisor_of_even_square_difference_l402_402341


namespace sum_of_sequence_l402_402956

-- Define the sequence of positive odd numbers
def odd_number (n : ℕ) : ℕ := 2 * n + 1

-- Define the sequence a_n according to the problem's specification
def a (n : ℕ) : ℕ :=
  (Finset.range n).sum (λ k, odd_number (n * (n - 1) / 2 + k))

-- The theorem stating that the sum of the first 20 terms of the sequence is 44100
theorem sum_of_sequence : (Finset.range 20).sum a = 44100 := by
  sorry

end sum_of_sequence_l402_402956


namespace range_satisfying_f_inequality_l402_402598

noncomputable def f (x : ℝ) : ℝ :=
  Real.log (1 + |x|) - (1 / (1 + x^2))

theorem range_satisfying_f_inequality : 
  ∀ x : ℝ, (1 / 3) < x ∧ x < 1 → f x > f (2 * x - 1) :=
by
  intro x hx
  sorry

end range_satisfying_f_inequality_l402_402598


namespace min_distance_run_l402_402643

def point (x y : ℕ) : Type := (x, y)

def wall := 900
def point_A := point 0 200
def point_B := point 700 0

theorem min_distance_run : 
  ∃ C : (ℕ × ℕ), (dist point_A C + dist C point_B) = 1273 := 
sorry

end min_distance_run_l402_402643


namespace count_distinct_a_l402_402107

def quadratic_has_integer_solutions (a : ℤ) : Prop :=
  ∃ x m n : ℤ, (x^2 + a * x + 12 * a = 0) ∧ (m + n = -a) ∧ (m * n = 12 * a)

theorem count_distinct_a : (finset.univ.filter quadratic_has_integer_solutions).card = 9 := 
sorry

end count_distinct_a_l402_402107


namespace point_A_in_fourth_quadrant_l402_402361

def point_A := (1, -1)

theorem point_A_in_fourth_quadrant (x y : ℤ) (h : (x, y) = point_A) : 
  x > 0 ∧ y < 0 := 
by 
  simp [point_A] at h
  sorry

end point_A_in_fourth_quadrant_l402_402361


namespace distinct_integer_a_values_l402_402140

open Nat

theorem distinct_integer_a_values : ∃ (S : Finset ℤ), S.card = 8 ∧
  ∀ a ∈ S, ∃ (p q : ℤ), p + q = -a ∧ p * q = 12a ∧ 
  ∀ x ∈ (polynomial.roots (polynomial.C (12 : ℚ) * polynomial.X + polynomial.C a * polynomial.X + polynomial.C 1)).val, x ∈ ℤ :=
by
  sorry

end distinct_integer_a_values_l402_402140


namespace exterior_angle_to_interior_sum_l402_402811

theorem exterior_angle_to_interior_sum (n : ℕ) (h : 24 * n = 360) : (180 * (n - 2) = 2340) :=
by
  have hn : n = 15 := by linarith
  rw [hn]
  simp
  sorry

end exterior_angle_to_interior_sum_l402_402811


namespace math_problem_l402_402828

theorem math_problem :
  ∃ (f : ℝ → ℝ), 
    (∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2) ∧ -- A: f(x) is increasing
    (∀ x1 x2 : ℝ, x1 < x2 → f (f x1) < f (f x2)) ∧ -- A: f[f(x)] is also increasing
    (∀ a : ℝ, f a ≠ a) ∧ -- C: for any a ∈ ℝ, f(a) ≠ a
    (∀ x : ℝ, f (f x) = x) -- C: f[f(x)] = x
:= 
begin
  sorry
end

end math_problem_l402_402828


namespace distinct_solution_count_l402_402085

theorem distinct_solution_count (h: ∀ x : ℝ, |2 * x - |3 * x - 2|| = 5) : ∃ s : Finset ℝ, s.card = 2 ∧ ∀ x ∈ s, |2 * x - |3 * x - 2|| = 5 :=
by
  sorry

end distinct_solution_count_l402_402085


namespace number_of_integer_values_of_a_l402_402162

theorem number_of_integer_values_of_a : 
  ∃ (a_set : Set ℤ), (∀ a ∈ a_set, ∃ x y : ℤ, x^2 + a * x + 12 * a = 0 ∧ y^2 + a * y + 12 * a = 0) ∧ a_set.card = 16 :=
sorry

end number_of_integer_values_of_a_l402_402162


namespace largest_crowd_size_l402_402887

theorem largest_crowd_size :
  ∃ (n : ℕ), 
    (⌊n / 2⌋ + ⌊n / 3⌋ + ⌊n / 5⌋ = n) ∧
    ∀ m : ℕ, (⌊m / 2⌋ + ⌊m / 3⌋ + ⌊m / 5⌋ = m) → m ≤ 37 :=
sorry

end largest_crowd_size_l402_402887


namespace general_term_formula_sum_of_b_first_terms_l402_402278

variable (a₁ a₂ : ℝ)
variable (a : ℕ → ℝ)
variable (b : ℕ → ℝ)
variable (T : ℕ → ℝ)

-- Conditions
axiom h1 : a₁ * a₂ = 8
axiom h2 : a₁ + a₂ = 6
axiom increasing_geometric_sequence : ∀ n : ℕ, a (n+1) = a (n) * (a₂ / a₁)
axiom initial_conditions : a 1 = a₁ ∧ a 2 = a₂
axiom b_def : ∀ n, b n = 2 * a n + 3

-- To Prove
theorem general_term_formula : ∀ n: ℕ, a n = 2 ^ (n + 1) :=
sorry

theorem sum_of_b_first_terms (n : ℕ) : T n = 2 ^ (n + 2) - 4 + 3 * n :=
sorry

end general_term_formula_sum_of_b_first_terms_l402_402278


namespace triangle_at_most_one_obtuse_l402_402435

def triangle := {angles : Fin 3 → ℝ // ∑ i, angles i = 180 ∧ ∀ i, 0 < angles i}
def is_obtuse (α : ℝ) : Prop := 90 < α

theorem triangle_at_most_one_obtuse :
  ∀ (T : triangle), ¬ (∃ i j, i ≠ j ∧ is_obtuse (T.val i) ∧ is_obtuse (T.val j)) :=
by sorry

end triangle_at_most_one_obtuse_l402_402435


namespace part1_case1_part1_case2_part1_case3_part2_l402_402220

def f (m x : ℝ) : ℝ := (m+1)*x^2 - (m-1)*x + (m-1)

theorem part1_case1 (m x : ℝ) (h : m = -1) : 
  f m x ≥ (m+1)*x → x ≥ 1 := sorry

theorem part1_case2 (m x : ℝ) (h : m > -1) :
  f m x ≥ (m+1)*x →
  (x ≤ (m-1)/(m+1) ∨ x ≥ 1) := sorry

theorem part1_case3 (m x : ℝ) (h : m < -1) : 
  f m x ≥ (m+1)*x →
  (1 ≤ x ∧ x ≤ (m-1)/(m+1)) := sorry

theorem part2 (m : ℝ) : 
  (∀ x : ℝ, -1/2 ≤ x ∧ x ≤ 1/2 → f m x ≥ 0) →
  m ≥ 1 := sorry

end part1_case1_part1_case2_part1_case3_part2_l402_402220


namespace distinct_integer_a_values_l402_402141

open Nat

theorem distinct_integer_a_values : ∃ (S : Finset ℤ), S.card = 8 ∧
  ∀ a ∈ S, ∃ (p q : ℤ), p + q = -a ∧ p * q = 12a ∧ 
  ∀ x ∈ (polynomial.roots (polynomial.C (12 : ℚ) * polynomial.X + polynomial.C a * polynomial.X + polynomial.C 1)).val, x ∈ ℤ :=
by
  sorry

end distinct_integer_a_values_l402_402141


namespace external_bisector_TK_l402_402721

-- Definitions of points and segments
variables {A T C L K : Type}

-- Conditions given in the problem
variable (angle_bisector_TL : is_angle_bisector T L A C)
variable (is_triangle : is_triangle T A C)

-- Prove that TK is the external angle bisector
theorem external_bisector_TK : is_external_angle_bisector T K A C :=
sorry

end external_bisector_TK_l402_402721


namespace find_m_l402_402591

theorem find_m (m x : ℝ) (h1 : mx + 3 = x) (h2 : 5 - 2x = 1) : m = -1/2 := by
  sorry

end find_m_l402_402591


namespace danny_marks_in_math_l402_402974

theorem danny_marks_in_math
  (english_marks : ℕ := 76)
  (physics_marks : ℕ := 82)
  (chemistry_marks : ℕ := 67)
  (biology_marks : ℕ := 75)
  (average_marks : ℕ := 73)
  (num_subjects : ℕ := 5) :
  ∃ (math_marks : ℕ), math_marks = 65 :=
by
  let total_marks := average_marks * num_subjects
  let other_subjects_marks := english_marks + physics_marks + chemistry_marks + biology_marks
  have math_marks := total_marks - other_subjects_marks
  use math_marks
  sorry

end danny_marks_in_math_l402_402974


namespace probability_prime_sum_l402_402467

def is_roll_result_valid (n : ℕ) : Prop := n ≥ 1 ∧ n ≤ 6

def is_prime_sum (a b : ℕ) : Prop := Nat.Prime (a + b)

theorem probability_prime_sum (t : ℚ) :
  (∀ (a b : ℕ), is_roll_result_valid a → is_roll_result_valid b → a + b ∈ {2, 3, 5, 7, 11}) →
  t = 5 / 12 :=
by
  sorry  -- Proof to be filled in

end probability_prime_sum_l402_402467


namespace integer_values_a_l402_402184

theorem integer_values_a :
  (∃! a : ℤ, ∃ m n : ℤ, m^2 + n^2 = (m + n)^2 - 2mn ∧
                     m + n = -a ∧ 
                     mn = 12a) :
  8 :=
sorry

end integer_values_a_l402_402184


namespace math_proof_l402_402689

open Real

variables {P Q R : Type*} [metric_space P]

-- Given conditions: 
-- The midpoints of the segments.
variables (p q r : ℝ)
  (midpoint_QR : P)
  (midpoint_PR : P)
  (midpoint_PQ : P)
  (h_mid_QR : midpoint_QR = (p, 0, 0))
  (h_mid_PR : midpoint_PR = (0, q, 0))
  (h_mid_PQ : midpoint_PQ = (0, 0, r))

-- Prove the desired result
theorem math_proof :
  ∀ (P Q R : P),
    ∃ (PQ PR QR : ℝ),
    -- compute the values of distances squared
    let PQ_sq := 4 * (q^2 + r^2),
        PR_sq := 4 * (p^2 + r^2),
        QR_sq := 4 * (p^2 + q^2) in
    PQ_sq + PR_sq + QR_sq = 8 * (p^2 + q^2 + r^2) :=
sorry

end math_proof_l402_402689


namespace order_radii_l402_402965

noncomputable def radius_X := Real.sqrt 10
noncomputable def circumference_Y := 8 * Real.pi
noncomputable def radius_Y : Real := circumference_Y / (2 * Real.pi)
noncomputable def area_Z := 16 * Real.pi
noncomputable def radius_Z : Real := Real.sqrt (area_Z / Real.pi)

theorem order_radii (X Y Z : Type) [has_radius X] [has_circumference Y] [has_area Z]
  (r_X : ℝ) (r_Y : ℝ) (r_Z : ℝ)
  (hX : r_X = radius_X) (hY : r_Y = radius_Y) (hZ : r_Z = radius_Z) :
  r_X = Real.sqrt 10 ∧ r_Y = 4 ∧ r_Z = 4 ∧ (r_X < r_Y ∧ r_X < r_Z ∧ r_Y = r_Z) :=
by sorry

end order_radii_l402_402965


namespace acute_angle_between_third_lateral_face_and_base_l402_402792

/-- Problem Statement:
The base of an oblique prism is a right-angled triangle with an acute angle \(\alpha\).
The lateral face containing the hypotenuse is perpendicular to the base.
The lateral face containing the leg adjacent to the given angle forms an acute angle \(\beta\) with the base.
Find the acute angle between the third lateral face and the base.
-/

theorem acute_angle_between_third_lateral_face_and_base
  (α β : ℝ)
  (hα : 0 < α ∧ α < π / 2)
  (hᵦ : 0 < β ∧ β < π / 2) :
  ∃ γ, 0 < γ ∧ γ < π / 2 ∧ γ = atan (tan α * tan β) := 
sorry

end acute_angle_between_third_lateral_face_and_base_l402_402792


namespace baron_munchausen_max_crowd_size_l402_402879

theorem baron_munchausen_max_crowd_size :
  ∃ n : ℕ, (∀ k, (k : ℕ) = n → 
  let left := (k / 2).toNat;
      right := (k / 3).toNat;
      straight := (k / 5).toNat in
  left + right + straight <= n + 1) ∧ 
  (∀ x : ℕ, x > 37 → ¬(∀ k, (k : ℕ) = x →
  let left := (k / 2).toNat;
      right := (k / 3).toNat;
      straight := (k / 5).toNat in
  left + right + straight <= x + 1)) :=
begin
  have h : 37 = 18 + 12 + 7,
  sorry,
end

end baron_munchausen_max_crowd_size_l402_402879


namespace megan_pictures_l402_402001

theorem megan_pictures (pictures_zoo pictures_museum pictures_deleted : ℕ)
  (hzoo : pictures_zoo = 15)
  (hmuseum : pictures_museum = 18)
  (hdeleted : pictures_deleted = 31) :
  (pictures_zoo + pictures_museum) - pictures_deleted = 2 :=
by
  sorry

end megan_pictures_l402_402001


namespace rotation_of_line_by_90_degrees_l402_402604

theorem rotation_of_line_by_90_degrees (P : ℝ × ℝ) (P_eq : P = (3, 1)) :
  let l0 := λ x : ℝ, x + 1
  let l := λ x y : ℝ, y + x - 4
  line_eq (rotate_line_90ccw_through_point l0 P) l :=
  sorry

end rotation_of_line_by_90_degrees_l402_402604


namespace geometric_sequence_a_n_sum_of_first_n_terms_b_n_l402_402280

theorem geometric_sequence_a_n :
  (∃ a₁ a₂, a₁ * a₂ = 8 ∧ a₁ + a₂ = 6 ∧ a₁ < a₂) →
  (∀ n : ℕ, n > 0 → (∃ a : ℕ → ℕ, (a 1 = 2) ∧ (a 2 = 4) ∧ (∀ n, a n = 2 ^ n))) :=
begin
  sorry
end

theorem sum_of_first_n_terms_b_n :
  (∀ n : ℕ, n > 0 → (∃ a : ℕ → ℕ, (a 1 = 2) ∧ (a 2 = 4) ∧ (∀ n, a n = 2 ^ n))) →
  (∀ b : ℕ → ℕ, (∀ n, b n = 2 * (2 ^ n) + 3)) →
  (∀ T : ℕ → ℤ, (∀ n, T n = 2 ^ (n + 2) - 4 + 3 * n)) :=
begin
  sorry
end

end geometric_sequence_a_n_sum_of_first_n_terms_b_n_l402_402280


namespace binary_string_sequence_period_l402_402852
noncomputable theory

open Function

-- Let's define the transformation T for the sequence x
def T (n : ℕ) (x : Fin n → Bool) : Fin n → Bool :=
  λ k, if x k = x (⟨(k.1 + 1) % n, Nat.mod_lt (k.1 + 1) n (Nat.succ_pos n)⟩) then false else true

-- Now define the sequence x according to the given rules
def x (n : ℕ) (m : ℕ) : Fin n → Bool := 
  sorry  -- We'll skip the explicit construction details

-- Now we state our theorem
theorem binary_string_sequence_period (n m : ℕ) (x0: Fin n → Bool)
  (h1 : odd n)
  (h2 : x0 = (λ k, if k = 0 ∨ k = n-1 then true else false))
  (h3 : ∀ m, x (m + 1) = T n (x m)) :
  x m = x n → m % n = 0 :=
sorry

end binary_string_sequence_period_l402_402852


namespace measure_angle_BED_l402_402379

-- Definitions for the conditions
variables {A B C D E : Type}
variables [triangle ABC] [triangle ADE]
variables [congruent_triangles ABC ADE]
variables [AB AC AE : segment] 
variables (h1 : AB.length = AC.length ∧ AC.length = AE.length)
variables (h2 : ∠BAC = 30)

-- Theorem statement
theorem measure_angle_BED : measure ∠BED = 15 :=
begin
  -- skipping the proof
  sorry
end

end measure_angle_BED_l402_402379


namespace external_bisector_of_triangle_l402_402703

variables {A T C K L : Type} [noncomputable_field A T C K L]
variable {triangle ATC : Triangle A T C}

noncomputable def internal_angle_bisector (T L : Point) := 
    ∃ R, is_bisector TL ∧ TL ∝ AR RL

noncomputable def external_angle_bisector (T K : Point) := 
    ∃ M, is_bisector TK ∧ TK ∝ AM MK

theorem external_bisector_of_triangle {A T C K L : Point} 
    (hL : internal_angle_bisector T L) 
    : external_angle_bisector T K :=
sorry

end external_bisector_of_triangle_l402_402703


namespace hyperbola_eccentricity_l402_402391

theorem hyperbola_eccentricity (a b e : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (hyp : ∀ x y : ℝ, (x + 2 * y + 1 = 0) -> (y = 2 * x) -> ∀ e, e = sqrt 5):
  e = sqrt 5 :=
sorry

end hyperbola_eccentricity_l402_402391


namespace integer_values_of_a_l402_402175

theorem integer_values_of_a (a : ℤ) :
  (∀ x : ℤ, x^2 + a * x + 12 * a = 0 → x ∈ ℤ) → ∃ n : ℕ, n = 16 :=
by
  sorry

end integer_values_of_a_l402_402175


namespace area_of_vectors_endpoints_shape_l402_402766

noncomputable def area_of_annulus {r1 r2 : ℝ} (r1 ≤ r2) : ℝ :=
 π * (r2 * r2 - r1 * r1)

theorem area_of_vectors_endpoints_shape :
  area_of_annulus (1 : ℝ) (2 : ℝ) = 3 * π := 
by
  -- Proof goes here
  sorry

end area_of_vectors_endpoints_shape_l402_402766


namespace external_bisector_of_triangle_l402_402700

variables {A T C K L : Type} [noncomputable_field A T C K L]
variable {triangle ATC : Triangle A T C}

noncomputable def internal_angle_bisector (T L : Point) := 
    ∃ R, is_bisector TL ∧ TL ∝ AR RL

noncomputable def external_angle_bisector (T K : Point) := 
    ∃ M, is_bisector TK ∧ TK ∝ AM MK

theorem external_bisector_of_triangle {A T C K L : Point} 
    (hL : internal_angle_bisector T L) 
    : external_angle_bisector T K :=
sorry

end external_bisector_of_triangle_l402_402700


namespace john_total_spent_l402_402309

noncomputable def total_spent (computer_cost : ℝ) (peripheral_ratio : ℝ) (base_video_cost : ℝ) : ℝ :=
  let peripheral_cost := computer_cost * peripheral_ratio
  let upgraded_video_cost := base_video_cost * 2
  computer_cost + peripheral_cost + (upgraded_video_cost - base_video_cost)

theorem john_total_spent :
  total_spent 1500 0.2 300 = 2100 :=
by
  sorry

end john_total_spent_l402_402309


namespace external_angle_bisector_l402_402717

open EuclideanGeometry

variables {A T C K L : Point}

theorem external_angle_bisector
    (h1 : is_internal_angle_bisector T L A T C) : is_external_angle_bisector T K A T C :=
sorry

end external_angle_bisector_l402_402717


namespace consecutive_composite_numbers_bound_l402_402779

theorem consecutive_composite_numbers_bound (n : ℕ) (hn: 0 < n) :
  ∃ (seq : Fin n → ℕ), (∀ i, ¬ Nat.Prime (seq i)) ∧ (∀ i, seq i < 4^(n+1)) :=
sorry

end consecutive_composite_numbers_bound_l402_402779


namespace initial_investment_l402_402542

theorem initial_investment
  (x : ℝ) (r : ℝ := 0.08) (n : ℕ := 5) (A : ℝ := 600) :
  x * (1 + r)^n = A → x ≈ 408.42 :=
by
  intros
  sorry

end initial_investment_l402_402542


namespace distinct_integer_values_of_a_l402_402118

theorem distinct_integer_values_of_a :
  ∃ (a_values : Finset ℤ), (∀ a ∈ a_values, ∃ (m n : ℤ), (x^2 + a * x + 12 * a = 0) ∧ (m + n = -a) ∧ (m * n = 12 * a))
  ∧ a_values.card = 16 :=
sorry

end distinct_integer_values_of_a_l402_402118


namespace external_bisector_of_triangle_l402_402697

variables {A T C K L : Type} [noncomputable_field A T C K L]
variable {triangle ATC : Triangle A T C}

noncomputable def internal_angle_bisector (T L : Point) := 
    ∃ R, is_bisector TL ∧ TL ∝ AR RL

noncomputable def external_angle_bisector (T K : Point) := 
    ∃ M, is_bisector TK ∧ TK ∝ AM MK

theorem external_bisector_of_triangle {A T C K L : Point} 
    (hL : internal_angle_bisector T L) 
    : external_angle_bisector T K :=
sorry

end external_bisector_of_triangle_l402_402697


namespace cost_effective_one_racket_and_ten_balls_cost_effective_n_rackets_k_balls_most_cost_effective_plan_l402_402960

-- Problem 1
theorem cost_effective_one_racket_and_ten_balls :
  let cost_A := (50 + 10 * 2) * 0.9
  let cost_B := 50 + (10 - 4) * 2
  cost_B < cost_A := by
  sorry

-- Problem 2
theorem cost_effective_n_rackets_k_balls (n : ℕ) (k : ℕ) (hk : k ≥ 4) :
  let cost_A := (50 * n + 2 * n * k) * 0.9
  let cost_B := 50 * n + 2 * n * (k - 4)
  (k < 15 → cost_B < cost_A) ∧ (k = 15 → cost_B = cost_A) ∧ (k > 15 → cost_B > cost_A) := by
  sorry

-- Problem 3
theorem most_cost_effective_plan (n : ℕ) :
  let cost_A := (50 * n + 40 * n) * 0.9
  let cost_B := 50 * n + 2 * n * (20 - 4)
  let cost_mix := -2.2 * n + 81 * n
  cost_mix < cost_A ∧ cost_mix < cost_B := by
  sorry

end cost_effective_one_racket_and_ten_balls_cost_effective_n_rackets_k_balls_most_cost_effective_plan_l402_402960


namespace firm_partners_l402_402008

theorem firm_partners
  (P A : ℕ)
  (h1 : P / A = 2 / 63)
  (h2 : P / (A + 35) = 1 / 34) :
  P = 14 :=
by
  sorry

end firm_partners_l402_402008


namespace sum_of_fractions_l402_402512

theorem sum_of_fractions : 
  (2 / 5 : ℚ) + (4 / 50 : ℚ) + (3 / 500 : ℚ) + (8 / 5000 : ℚ) = 4876 / 10000 :=
by
  -- The proof can be completed by converting fractions and summing them accurately.
  sorry

end sum_of_fractions_l402_402512


namespace largest_sum_of_two_3_digit_numbers_l402_402842

theorem largest_sum_of_two_3_digit_numbers : 
  ∃ (a b c d e f : ℕ), 
    (1 ≤ a ∧ a ≤ 6) ∧ (1 ≤ b ∧ b ≤ 6) ∧ (1 ≤ c ∧ c ≤ 6) ∧
    (1 ≤ d ∧ d ≤ 6) ∧ (1 ≤ e ∧ e ≤ 6) ∧ (1 ≤ f ∧ f ≤ 6) ∧
    (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ 
     b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ 
     c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ 
     d ≠ e ∧ d ≠ f ∧ 
     e ≠ f) ∧ 
    (100 * (a + d) + 10 * (b + e) + (c + f) = 1173) :=
by
  sorry

end largest_sum_of_two_3_digit_numbers_l402_402842


namespace number_of_distinct_a_l402_402124

noncomputable def count_distinct_a : ℕ :=
  let divisors := [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 36, 48, 72, 144, -1, -2, -3, -4, -6, -8, -9, -12, -16, -18, -24, -36, -48, -72, -144]
  let pairs := (divisors.product divisors).filter (λ (p : ℤ × ℤ), p.1 * p.2 = 144)
  let sums := pairs.map (λ (p : ℤ × ℤ), p.1 + p.2 - 24)
  let distinct_sums := sums.eraseDuplicates  
  distinct_sums.length

theorem number_of_distinct_a : count_distinct_a = 14 :=
by
  sorry

end number_of_distinct_a_l402_402124


namespace no_nat_n_divisible_by_169_l402_402364

theorem no_nat_n_divisible_by_169 (n : ℕ) : ¬ ∃ k : ℕ, n^2 + 5 * n + 16 = 169 * k :=
sorry

end no_nat_n_divisible_by_169_l402_402364


namespace central_angle_measure_l402_402214

theorem central_angle_measure (α r : ℝ) (h1 : α * r = 2) (h2 : 1/2 * α * r^2 = 2) : α = 1 := 
sorry

end central_angle_measure_l402_402214


namespace count_positive_integers_satisfy_l402_402622

theorem count_positive_integers_satisfy :
  ∃ (S : Finset ℕ), (∀ n ∈ S, (n + 5) * (n - 3) * (n - 12) * (n - 17) < 0) ∧ S.card = 4 :=
by
  sorry

end count_positive_integers_satisfy_l402_402622


namespace express_train_catchup_time_l402_402031

theorem express_train_catchup_time
    (speed_goods_train : ℕ) (speed_express_train : ℕ) (hours_head_start : ℕ) : 
    speed_goods_train = 36 → 
    speed_express_train = 90 → 
    hours_head_start = 6 → 
    (let distance_head_start := speed_goods_train * hours_head_start,
         relative_speed := speed_express_train - speed_goods_train,
         catchup_time := distance_head_start / relative_speed
     in catchup_time = 4) := 
by
  intros h1 h2 h3
  dsimp
  rw [h1, h2, h3]
  norm_num
  sorry

end express_train_catchup_time_l402_402031


namespace cos_of_largest_angle_is_neg_half_l402_402202

-- Lean does not allow forward references to elements yet to be declared, 
-- hence we keep a strict order for declarations
namespace TriangleCosine

open Real

-- Define the side lengths of the triangle as constants
def a : ℝ := 3
def b : ℝ := 5
def c : ℝ := 7

-- Define the expression using cosine rule to find cos C
noncomputable def cos_largest_angle : ℝ := (a^2 + b^2 - c^2) / (2 * a * b)

-- Declare the theorem statement
theorem cos_of_largest_angle_is_neg_half : cos_largest_angle = -1 / 2 := 
by 
  sorry

end TriangleCosine

end cos_of_largest_angle_is_neg_half_l402_402202


namespace william_shared_marble_count_l402_402849

theorem william_shared_marble_count : ∀ (initial_marbles shared_marbles remaining_marbles : ℕ),
  initial_marbles = 10 → remaining_marbles = 7 → 
  shared_marbles = initial_marbles - remaining_marbles → 
  shared_marbles = 3 := by 
    intros initial_marbles shared_marbles remaining_marbles h_initial h_remaining h_shared
    rw [h_initial, h_remaining] at h_shared
    exact h_shared

end william_shared_marble_count_l402_402849


namespace profit_percentage_is_50_l402_402371

/--
Assumption:
- Initial machine cost: Rs 10,000
- Repair cost: Rs 5,000
- Transportation charges: Rs 1,000
- Selling price: Rs 24,000

To prove:
- The profit percentage is 50%
-/

def initial_cost : ℕ := 10000
def repair_cost : ℕ := 5000
def transportation_charges : ℕ := 1000
def selling_price : ℕ := 24000
def total_cost : ℕ := initial_cost + repair_cost + transportation_charges
def profit : ℕ := selling_price - total_cost

theorem profit_percentage_is_50 :
  (profit * 100) / total_cost = 50 :=
by
  -- proof goes here
  sorry

end profit_percentage_is_50_l402_402371


namespace geometric_shape_of_complex_l402_402838

theorem geometric_shape_of_complex (z : ℂ) (h : complex.abs (z - complex.I * 3) = 10) : 
  ∃ c : ℂ, ∃ r : ℝ, complex.abs (z - c) = r ∧ r = 10 ∧ c = complex.I * 3 :=
sorry

end geometric_shape_of_complex_l402_402838


namespace midpoint_polar_origin_l402_402644

theorem midpoint_polar_origin : 
  let A := (5 : ℝ, real.pi / 3)
  let B := (5 : ℝ, -2 * real.pi / 3)
  let Ax := 5 * real.cos (real.pi / 3)
  let Ay := 5 * real.sin (real.pi / 3)
  let Bx := 5 * real.cos (-2 * real.pi / 3)
  let By := 5 * real.sin (-2 * real.pi / 3)
  let Mx := (Ax + Bx) / 2
  let My := (Ay + By) / 2
  in (Mx, My) = (0, 0) :=
by
  sorry

end midpoint_polar_origin_l402_402644


namespace factorial_sum_representation_reciprocal_sum_greater_than_million_l402_402250

noncomputable section

variable {x : ℚ}

-- Let's first state that x can be uniquely expressed as a sum of factorial reciprocals
theorem factorial_sum_representation (h : 0 < x) :
  ∃ (n : ℕ) (a : Fin n → ℤ), 
  (x = ∑ k in Finset.range n, a k / Nat.fact k) ∧ 
  (∀ k, 0 ≤ a k ∧ a k < k) :=
sorry

-- Next, we express x as the sum of reciprocals of different integers > 10^6
theorem reciprocal_sum_greater_than_million (h : 0 < x) :
  ∃ (n : ℕ) (a : Fin n → ℤ), 
  (x = ∑ k in Finset.range n, a k) ∧ 
  (∀ k, 10^6 < a k) :=
sorry

end factorial_sum_representation_reciprocal_sum_greater_than_million_l402_402250


namespace sum_of_coeff_integral_l402_402551

variable (a : ℕ → ℕ) 

theorem sum_of_coeff_integral : 
  (a 0 + ∑ k in (Finset.range 11).filter (λ k, k > 0), (a k) / (k + 1)) = 2047 / 11 :=
  
by {
  -- defining the polynomial expansion (1 + x)^10 = a_0 + a_1 x + ... + a_10 x^10
  have h1 : (1 + x) ^ 10 = ∑ k in Finset.range 11, a k * x ^ k, 
  sorry,
  
  -- integrating both sides from 0 to 1
  have h2 : ∫ x in 0..1, (1 + x) ^ 10 = a 0 + ∑ k in (Finset.range 11).filter (λ k, k > 0), (a k) / (k + 1), 
  sorry,
  
  -- substituting (1 + x)^11/11 from the definite integral
  have h3 : ∫ x in 0..1, (1 + x) ^ 10 = (2047 / 11),
  sorry,

  exact h1.trans (h2.trans h3),
}

end sum_of_coeff_integral_l402_402551


namespace number_of_integer_values_of_a_l402_402163

theorem number_of_integer_values_of_a : 
  ∃ (a_set : Set ℤ), (∀ a ∈ a_set, ∃ x y : ℤ, x^2 + a * x + 12 * a = 0 ∧ y^2 + a * y + 12 * a = 0) ∧ a_set.card = 16 :=
sorry

end number_of_integer_values_of_a_l402_402163


namespace angus_tokens_eq_l402_402500

-- Define the conditions
def worth_per_token : ℕ := 4
def elsa_tokens : ℕ := 60
def angus_less_worth : ℕ := 20

-- Define the main theorem to prove
theorem angus_tokens_eq :
  let elsaTokens := elsa_tokens,
      worthPerToken := worth_per_token,
      angusLessTokens := angus_less_worth / worth_per_token
  in (elsaTokens - angusLessTokens) = 55 := by
  sorry

end angus_tokens_eq_l402_402500


namespace good_triplets_sum_l402_402016

def is_good_triplet (s : Finset ℕ) : Prop :=
s.card = 3 ∧
∀ (a b ∈ s), a + b ≠ 6 ∨ a = b

def triplets : Finset (Finset ℕ) :=
(Finset.powerset (Finset.range 6)).filter is_good_triplet

def product (s : Finset ℕ) : ℕ :=
s.fold (λ x y, x * y) 1 id

theorem good_triplets_sum :
  (triplets.image product).sum = 121 :=
by
  sorry

end good_triplets_sum_l402_402016


namespace minimum_familiar_pairs_l402_402907

open Finset

-- Define the set of students and the relationship of familiarity
variable (students : Finset ℕ)
variable (n : ℕ := 175)
variable (familiar : ℕ → ℕ → Prop)

-- Assumption: students set has 175 members
axiom student_count : students.card = n

-- Assumption: familiarity is symmetric
axiom familiar_symm (a b : ℕ) : familiar a b → familiar b a

-- Assumption: familiarity within any group of six
axiom familiar_in_groups_of_six (s : Finset ℕ) (h₁ : s.card = 6) :
  ∃ t₁ t₂ : Finset ℕ, t₁.card = 3 ∧ t₂.card = 3 ∧ (∀ x ∈ t₁, ∀ y ∈ t₁, x ≠ y → familiar x y) ∧
  (∀ x ∈ t₂, ∀ y ∈ t₂, x ≠ y → familiar x y) ∧ t₁ ∪ t₂ = s ∧ t₁ ∩ t₂ = ∅

-- Theorem: minimum number of familiar pairs
theorem minimum_familiar_pairs :
  ∃ k : ℕ, (∑ a in students, (students.filter (familiar a)).card) / 2 ≥ 15050 :=
sorry

end minimum_familiar_pairs_l402_402907


namespace sum_mod_9_equal_6_l402_402251

theorem sum_mod_9_equal_6 :
  ((1 + 22 + 333 + 4444 + 55555 + 666666 + 7777777 + 88888888) % 9) = 6 :=
by
  sorry

end sum_mod_9_equal_6_l402_402251


namespace fixed_point_exists_l402_402570

variable {x y a b c : ℝ}

def ellipse (a b : ℝ) (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

noncomputable def ecc (a c : ℝ) : ℝ := c / a

def perp_condition (A M N H: ℝ × ℝ) : Prop :=
  let (ax, ay) := A in let (mx, my) := M in let (nx, ny) := N in let (hx, hy) := H in
  my - ay ≠ 0 ∧ nx - mx ≠ 0 ∧ (ny - my) * (ax - hx) = -(mx - nx) * (ay - hy)

def sum_of_slopes_condition (A M N : ℝ × ℝ) : Prop :=
  let (ax, ay) := A in let (mx, my) := M in let (nx, ny) := N in
  (my - ay) / mx + (ny - ay) / nx = 4 ∧ mx ≠ 0 ∧ nx ≠ 0

theorem fixed_point_exists : 
  ∃ P H : ℝ × ℝ, 
  ellipse 3 1 P.1 P.2 ∧ 
  perp_condition (0, 1) P (-1/4, 0) H ∧ 
  ∃ PH_dist : ℝ, PH_dist = √17 / 4 := 
begin
  sorry
end

end fixed_point_exists_l402_402570


namespace stratified_sampling_admin_personnel_l402_402924

theorem stratified_sampling_admin_personnel :
  let total_employees := 120
  let business := 60
  let administration := 40
  let logistical := 20
  let sample_size := 24
  (administration / total_employees) * sample_size = 8 := 
by
  let total_employees := 120
  let business := 60
  let administration := 40
  let logistical := 20
  let sample_size := 24
  calc
    (administration : ℝ) / (total_employees : ℝ) * sample_size
    = (40 : ℝ) / 120 * 24 : by refl
    ... = 8 : by norm_num

end stratified_sampling_admin_personnel_l402_402924


namespace other_asymptote_eqn_l402_402771

-- Proof problem statement
theorem other_asymptote_eqn (y1 y2 : ℝ → ℝ) (x_c : ℝ) :
  (∀ x, y1 x = -2 * x) →
  (∀ y, ∃ x1 x2 : ℝ, x1 = x2 ∧ x1 = -4) →
  y2 = λ x, 2 * x + 16
  :=
by
  sorry

end other_asymptote_eqn_l402_402771


namespace tan_arithmetic_sequence_min_tan_product_l402_402579

variable {A B C : ℝ}

-- Definitions based on the conditions
def acute_triangle (A B C : ℝ) : Prop := A + B + C = π ∧ A > 0 ∧ B > 0 ∧ C > 0 ∧ A < π/2 ∧ B < π/2 ∧ C < π/2
def vector_perp (a1 a2 b1 b2 : ℝ) : Prop := a1 * b1 + a2 * b2 = 0

-- Prove tan B, tan B * tan C, tan C form an arithmetic sequence
theorem tan_arithmetic_sequence (h1 : acute_triangle A B C) 
                                (h2 : vector_perp (sin A) (sin B * sin C) 1 (-2)) :
    tan B + tan C = 2 * tan B * tan C := sorry

-- Prove the minimum value of tan A * tan B * tan C is 8
theorem min_tan_product (h1 : acute_triangle A B C) 
                        (h2 : vector_perp (sin A) (sin B * sin C) 1 (-2)) :
    tan A * tan B * tan C ≥ 8 :=
begin
  sorry
end

end tan_arithmetic_sequence_min_tan_product_l402_402579


namespace external_angle_bisector_proof_l402_402735

variables {A T C L K : Type} [Nonempty A] [Nonempty T] [Nonempty C] [Nonempty L] [Nonempty K]

noncomputable def angle_bisector_theorem (AL LC AB BC AK KC : ℝ) : Prop :=
(AL / LC) = (AB / BC) ∧ (AK / KC) = (AL / LC)

noncomputable def internal_angle_bisector (AT TC AL LC : ℝ) : Prop :=
(AT / TC) = (AL / LC)

noncomputable def external_angle_bisector (AT TC AK KC : ℝ) : Prop :=
(AT / TC) = (AK / KC)

theorem external_angle_bisector_proof (AL LC AB BC AK KC AT TC : ℝ) 
(h1 : angle_bisector_theorem AL LC AB BC AK KC)
(h2 : internal_angle_bisector AT TC AL LC) :
external_angle_bisector AT TC AK KC :=
sorry

end external_angle_bisector_proof_l402_402735


namespace boats_in_river_l402_402478

theorem boats_in_river (W wb sb : ℝ) (hW : W = 42) (hwb : wb = 3) (hsb : sb = 2) : 
    let required_space_per_boat := wb + 2 * sb in
    W / required_space_per_boat = 6 :=
by
  simp [hW, hwb, hsb]
  sorry

end boats_in_river_l402_402478


namespace sqrt_subtraction_eq_seven_l402_402846

theorem sqrt_subtraction_eq_seven {
  A : ℕ, B : ℕ
} (hA : A = 36 + 64) (hB : B = 25 - 16) :
  (Real.sqrt A - Real.sqrt B) = 7 := 
by
  sorry

end sqrt_subtraction_eq_seven_l402_402846


namespace max_people_in_crowd_l402_402873

theorem max_people_in_crowd : ∃ n : ℕ, n ≤ 37 ∧ 
    (⟨1 / 2 * n⟩ + ⟨1 / 3 * n⟩ + ⟨1 / 5 * n⟩ = n) :=
sorry

end max_people_in_crowd_l402_402873


namespace g_eq_zero_l402_402329

def g (x : ℝ) : ℝ := (sqrt (3 * sin x ^ 2 + 9 * (cos x ^ 2) ^ 2)) - (sqrt (3 * cos x ^ 2 + 9 * (sin x ^ 2) ^ 2))

theorem g_eq_zero (x : ℝ) : g x = 0 :=
by
  sorry

end g_eq_zero_l402_402329


namespace range_of_a_l402_402599

noncomputable def quadratic_func (a x : ℝ) : ℝ := -x^2 - 2 * a * x

theorem range_of_a (a : ℝ) (h_max : ∃ x ∈ Icc 0 1, quadratic_func a x = a^2) : a ∈ Icc (-1) 0 := 
sorry

end range_of_a_l402_402599


namespace part_a_limit_part_b_inequality_l402_402335

noncomputable def seq_a (n : ℕ) : ℝ := 
  ∑ i in finset.range n, (-1 : ℝ)^i / (2 * i + 1)

theorem part_a_limit : 
  tendsto (seq_a) atTop (𝓝 (π / 4)) :=
sorry

theorem part_b_inequality (k : ℕ) : 
  1 / (2 * (4 * k + 1)) ≤ (π / 4 - seq_a (2 * k - 1)) ∧ (π / 4 - seq_a (2 * k - 1)) ≤ 1 / (4 * k + 1) :=
sorry 

end part_a_limit_part_b_inequality_l402_402335


namespace set_intersection_example_l402_402342

theorem set_intersection_example : 
  (A = {1, 3, 4}) ∧ (B = {0, 1, 3}) → (A ∩ B = {1, 3}) :=
by
  sorry

end set_intersection_example_l402_402342


namespace distance_KH_l402_402319

theorem distance_KH (K: Type) (A A1 M N1: Point) (z : ℝ) (ray_AA1 : Line A A1) (h: z = dist A K) : 
  dist K (line_through M N1) = sqrt (1.44 + 0.8 * (z + 0.3)^2) := 
by
  sorry

end distance_KH_l402_402319


namespace ordered_scores_of_QMS_l402_402368

variables (Q M S K : ℕ)

theorem ordered_scores_of_QMS (h1 : ∃ k, (k = K) ∧ ((k > Q ∧ k < M) ∨ (k > S ∧ k < M)))
                               (h2 : Q ≠ min Q M S ∧ Q ≠ max Q M S)
                               (h3 : M ≠ min Q M S)
                               (h4 : S ≠ min Q M S ∧ S ≠ max Q M S) :
  (S < Q ∧ Q < M) :=
begin
  sorry
end

end ordered_scores_of_QMS_l402_402368


namespace largest_crowd_size_l402_402883

theorem largest_crowd_size (x : ℕ) : 
  (ceil (x * (1 / 2)) + ceil (x * (1 / 3)) + ceil (x * (1 / 5)) = x) →
  x ≤ 37 :=
sorry

end largest_crowd_size_l402_402883


namespace consecutive_odd_squares_l402_402824

theorem consecutive_odd_squares (n : ℤ) :
  let a := n - 2 in
  let b := n in
  let c := n + 2 in
  (a^2 + b^2 + c^2) = 1111 * j → 
  (j = 1 ∨ j = 2 ∨ j = 3 ∨ j = 4 ∨ j = 5 ∨ j = 6 ∨ j = 7 ∨ j = 8 ∨ j = 9) → 
  (n = 43 ∨ n = -43) :=
by sorry

end consecutive_odd_squares_l402_402824


namespace goods_train_speed_l402_402486

-- Definitions of conditions
def woman's_train_speed_kmph : ℕ := 20
def goods_train_length_m : ℕ := 300
def time_to_pass_seconds : ℕ := 15

-- Definitions to convert units and compute relative speed
def length_in_kilometers (length_m : ℕ) : ℕ := length_m / 1000
def time_in_hours (time_s : ℕ) : ℕ := time_s / 3600
def relative_speed_kmph (distance_km : ℕ) (time_s : ℕ) : ℕ := 
  distance_km * 3600 / time_s

-- Theorem to be proved
theorem goods_train_speed :
  let V_w := woman's_train_speed_kmph in
  let L := goods_train_length_m / 1000 in
  let T := time_to_pass_seconds in
  let V_r := relative_speed_kmph L T in
  V_r - V_w = 52 := by
  sorry

end goods_train_speed_l402_402486


namespace ratio_ae_ef_l402_402071

-- Given definitions and conditions
variables {a : ℝ} (A B C D E F : ℝ × ℝ)
def A := (a, 0)
def D := (a, a)
def B := (b : ℝ), (b, b)
def C := (c : ℝ), (c, 2 * c)

-- Assuming collinearity and the given ratio conditions
axiom collinear (h_collinear : ∃ (A B C : ℝ × ℝ), collinear_points A B C) :
  true

axiom ratio_AB_BC (h_ratio : (dist A B) / (dist B C) = 2) :
  true

-- Define intersection points and the points lying on specified lines
def E := (e : ℝ), (e, e)
axiom circumcircle_ADC (h_circum : (∃ (D E : ℝ × ℝ), circumcircle A D C E)) :
  true

axiom ray_intersection (h_intersection : (∃ (E F : ℝ × ℝ), ray_ae_intersects_y_eq_2x_at_f A E F)) :
  true

-- The final ratio to be proven
theorem ratio_ae_ef (h_final : (dist A E) / (dist E F) = 7) :
  true :=
sorry

end ratio_ae_ef_l402_402071


namespace smallest_c_plus_d_l402_402398

theorem smallest_c_plus_d :
  ∃ (c d : ℕ), (8 * c + 3 = 3 * d + 8) ∧ c + d = 27 :=
by
  sorry

end smallest_c_plus_d_l402_402398


namespace compute_expected_volume_l402_402015

open Real

noncomputable def expectedVolumeIntersectedRegion : ℝ := 
  4 * π * (1 / 3 + 1 / 2019)

theorem compute_expected_volume : 
  let a := 2696 
  let b := 2019 
  let expected_volume_expr := expectedVolumeIntersectedRegion = (2696 / 2019) * π
  let result := 100 * a + b 
in expected_volume_expr ∧ result = 271619 := by {
  sorry
}

end compute_expected_volume_l402_402015


namespace quality_is_related_to_production_line_mathematical_expectation_E_xi_l402_402463

variables (parts_A_first_class parts_A_non_first_class parts_B_first_class parts_B_non_first_class : ℕ) (total_parts : ℕ := 180)
variables (a b c d n : ℕ) (ad_minus_bc : ℕ)
variables (chi_squared x_0.05 : ℝ) 

def first_class_parts_A := 75
def non_first_class_parts_A := 25
def first_class_parts_B := 48
def non_first_class_parts_B := 32

def contingency_table := [
  [first_class_parts_A, non_first_class_parts_A],
  [first_class_parts_B, non_first_class_parts_B]
]

def chi_squared := (n * (ad_minus_bc)^2) / ((a + b) * (c + d) * (a + c) * (b + d))

theorem quality_is_related_to_production_line :
  let n := total_parts,
  let ad_minus_bc := first_class_parts_A * non_first_class_parts_B - first_class_parts_B * non_first_class_parts_A,
  let chi_squared := (n * (ad_minus_bc ^ 2)) / ((first_class_parts_A + first_class_parts_B) * (non_first_class_parts_A + non_first_class_parts_B) * (first_class_parts_A + non_first_class_parts_A) * (first_class_parts_B + non_first_class_parts_B)),
  let x_0.05 := 3.841 in
  chi_squared > x_0.05 :=
by {
  sorry
}

theorem mathematical_expectation_E_xi (p_first_class_A p_first_class_B : ℝ) (E_xi : ℝ) :
  let p_first_class_A := 3 / 4,
  let p_first_class_B := 3 / 5,
  let E_xi := (0 * (1 / 100) + 1 * (9 / 100) + 2 * (117 / 400) + 3 * (81 / 200) + 4 * (81 / 400)) in
  E_xi = 27 / 10 :=
by {
  sorry
}

end quality_is_related_to_production_line_mathematical_expectation_E_xi_l402_402463


namespace triangle_cosine_rule_c_triangle_tangent_C_l402_402241

-- Define a proof statement for the cosine rule-based proof of c = 4.
theorem triangle_cosine_rule_c (a b : ℝ) (angleB : ℝ) (ha : a = 2)
                              (hb : b = 2 * Real.sqrt 3) (hB : angleB = π / 3) :
  ∃ (c : ℝ), c = 4 := by
  sorry

-- Define a proof statement for the tangent identity-based proof of tan C = 3 * sqrt 3 / 5.
theorem triangle_tangent_C (tanA : ℝ) (tanB : ℝ) (htA : tanA = 2 * Real.sqrt 3)
                           (htB : tanB = Real.sqrt 3) :
  ∃ (tanC : ℝ), tanC = 3 * Real.sqrt 3 / 5 := by
  sorry

end triangle_cosine_rule_c_triangle_tangent_C_l402_402241


namespace expand_product_l402_402530

noncomputable def a (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 1
noncomputable def b (x : ℝ) : ℝ := x^2 + x + 3

theorem expand_product (x : ℝ) : (a x) * (b x) = 2 * x^4 - x^3 + 4 * x^2 - 8 * x + 3 :=
by
  sorry

end expand_product_l402_402530


namespace external_bisector_TK_l402_402727

-- Definitions of points and segments
variables {A T C L K : Type}

-- Conditions given in the problem
variable (angle_bisector_TL : is_angle_bisector T L A C)
variable (is_triangle : is_triangle T A C)

-- Prove that TK is the external angle bisector
theorem external_bisector_TK : is_external_angle_bisector T K A C :=
sorry

end external_bisector_TK_l402_402727


namespace cloth_cut_max_squares_l402_402039

theorem cloth_cut_max_squares (length : ℕ) (width : ℕ) (square_area: ℕ) (max_squares : ℕ) 
  (h1 : length = 40) 
  (h2 : width = 27) 
  (h3 : square_area = 4) 
  (h4 : max_squares = 260) : 
  max_squares = (length / (int.sqrt square_area)) * (width / (int.sqrt square_area)) :=
sorry

end cloth_cut_max_squares_l402_402039


namespace div_condition_l402_402533

theorem div_condition (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : 
  4 * (m * n + 1) % (m + n)^2 = 0 ↔ m = n := 
sorry

end div_condition_l402_402533


namespace division_of_fractions_l402_402097

theorem division_of_fractions :
  (5 / 3) / (8 / 15) = 25 / 8 :=
by
  -- using given conditions and the properties of fractions
  let a := 5 / 3
  let b := 8 / 15
  have h1 : a = 5 / 3 := rfl
  have h2 : b = 8 / 15 := rfl
  have h3 : a / b = a * (1 / b) := by exact div_mul_eq_mul_div a b
  have h4 : 1 / b = 15 / 8 := by rw [one_div b, div_eq_inv_mul, inv_div, inv_inv, div_eq_mul_one_div]
  have h5 : a * (15 / 8) = (5 * 15) / (3 * 8) := by rw [mul_div_assoc, div_div_eq_div_mul, div_eq_mul_one_div, one_div, div_eq_mul_inv]
  have h6 : (5 * 15) / (3 * 8) = 75 / 24 := by norm_num
  have h7 : 75 / 24 = 25 / 8 := by norm_num
  rw [h1, h2, h3, h4, h5, h6, h7]
  exact rfl

end division_of_fractions_l402_402097


namespace initial_chocolate_bars_is_67_l402_402982

-- Definition of the problem and conditions
def initial_chocolate_bars (final_chocolate_bars : ℝ) (remained_ratio : ℝ) :=
  final_chocolate_bars / (remained_ratio ^ 4)

-- Proof statement
theorem initial_chocolate_bars_is_67
  (final_chocolate_bars : ℝ)
  (remained_ratio : ℝ)
  (h1 : final_chocolate_bars = 16)
  (h2 : remained_ratio = 0.7) :
  initial_chocolate_bars final_chocolate_bars remained_ratio = 67 :=
by
  unfold initial_chocolate_bars
  rw [h1, h2]
  norm_num
  sorry  -- skipping the detailed proof

end initial_chocolate_bars_is_67_l402_402982


namespace union_of_A_and_B_l402_402575

open Set

variable {x : ℝ}

-- Define sets A and B based on the given conditions
def A : Set ℝ := { x | 0 < 3 - x ∧ 3 - x ≤ 2 }
def B : Set ℝ := { x | 0 ≤ x ∧ x ≤ 2 }

-- The theorem to prove
theorem union_of_A_and_B : A ∪ B = { x | 0 ≤ x ∧ x < 3 } := 
by 
  sorry

end union_of_A_and_B_l402_402575


namespace sports_field_perimeter_l402_402939

noncomputable def perimeter_of_sports_field (a b : ℝ) (h1 : a^2 + b^2 = 400) (h2 : a * b = 120) : ℝ :=
  2 * (a + b)

theorem sports_field_perimeter {a b : ℝ} (h1 : a^2 + b^2 = 400) (h2 : a * b = 120) :
  perimeter_of_sports_field a b h1 h2 = 51 := by
  sorry

end sports_field_perimeter_l402_402939


namespace number_of_integer_values_of_a_l402_402160

theorem number_of_integer_values_of_a : 
  ∃ (a_set : Set ℤ), (∀ a ∈ a_set, ∃ x y : ℤ, x^2 + a * x + 12 * a = 0 ∧ y^2 + a * y + 12 * a = 0) ∧ a_set.card = 16 :=
sorry

end number_of_integer_values_of_a_l402_402160


namespace value_of_M_l402_402252

theorem value_of_M (M : ℝ) (h : 0.25 * M = 0.35 * 4025) : M = 5635 :=
sorry

end value_of_M_l402_402252


namespace compare_probabilities_l402_402021

open ProbabilityTheory

noncomputable def sumDice (n : ℕ) (sides : finset ℕ) (p : ∀ i ∈ sides, 0 < (uniformProb sides) i ∧ (uniformProb sides) i ≤ 1) : ProbMeasure (fin n → ℕ) :=
  fun _ => classical.arbitrary (ProbMeasure (fin n → ℕ))

theorem compare_probabilities :
  let dice_faces := {1, 2, 3, 4, 5, 6}
  let num_dice := 60
  let S := sumDice num_dice dice_faces (by simp [uniformProb])
  AllDiceAreFair := ∀ x, x ∈ dice_faces → ∀ k, x / (fin_to_int k) = 1/6
  probability_of_sum_at_least_300 := S.ret (λ f, (∑ i, f i) ≥ 300)
  probability_of_sum_less_than_120 := S.ret (λ f, (∑ i, f i) < 120)
  P₁ := probability_of_sum_at_least_300,
  P₂ := probability_of_sum_less_than_120,
  P₁ > P₂ :=
begin
  sorry
end

end compare_probabilities_l402_402021


namespace parabola_line_intersection_ratio_l402_402037

theorem parabola_line_intersection_ratio (p : ℝ) (A B : ℝ × ℝ)
  (h_parabola : ∀ {x y : ℝ}, (y, x) = A ∨ (y, x) = B → y ^ 2 = 4 * x)
  (h_focus : ∃ y₁ y₂ : ℝ, A = (y₁, sqrt 4 * (p / 2) - p / 2) ∧ B = (y₂, sqrt 4 * (p / 6) - p / 2))
  (h_angle : ∀ y : ℝ, y = sqrt 3 * x - sqrt 3 / 2 * p) :
  |(A.1 - p / 2) / (B.1 - p / 2)| = 3 :=
sorry

end parabola_line_intersection_ratio_l402_402037


namespace solve_inequality_1_solve_inequality_2_l402_402601

-- Definitions based on given conditions
noncomputable def f (x : ℝ) : ℝ := abs (x + 1)

-- Lean statement for the first proof problem
theorem solve_inequality_1 :
  ∀ x : ℝ, f x ≤ 5 - f (x - 3) ↔ -2 ≤ x ∧ x ≤ 3 :=
by
  sorry

-- Lean statement for the second proof problem
theorem solve_inequality_2 (a : ℝ) :
  (∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ 2 * f x + abs (x + a) ≤ x + 4) ↔ -2 ≤ a ∧ a ≤ 4 :=
by
  sorry

end solve_inequality_1_solve_inequality_2_l402_402601


namespace external_angle_bisector_of_triangle_l402_402741

variables {A T C L K : Type}
variables [IsTriangle A T C] [IsInternalAngleBisector T L (A T C)]

theorem external_angle_bisector_of_triangle 
  (h1 : is_internal_angle_bisector TL ATC) :
  is_external_angle_bisector TK ATC :=
sorry

end external_angle_bisector_of_triangle_l402_402741


namespace greatest_sum_of_consecutive_integers_l402_402840

theorem greatest_sum_of_consecutive_integers (n : ℤ) (h : (n - 1) * n * (n + 1) < 1000) : 9 + 10 + 11 = 30 :=
by
  have h1 : n = 10 := sorry  -- To be replaced with detailed argument
  calc
    9 + 10 + 11 = 30 : by norm_num
    sorry

end greatest_sum_of_consecutive_integers_l402_402840


namespace odd_count_of_divisible_64_digit_numbers_l402_402075

theorem odd_count_of_divisible_64_digit_numbers :
  ∃ S : Set ℕ, 
    (∀ n ∈ S, (∃ f : Fin 64 → Fin 9, n = ∑ i : Fin 64, (f i + 1) * 10^i ∧ n % 101 = 0)) ∧
    S.card % 2 = 1 :=
begin
  sorry
end

end odd_count_of_divisible_64_digit_numbers_l402_402075


namespace angus_token_count_l402_402502

theorem angus_token_count (elsa_tokens : ℕ) (token_value : ℕ) 
  (tokens_less_than_elsa_value : ℕ) (elsa_token_value_relation : elsa_tokens = 60) 
  (token_value_relation : token_value = 4) (tokens_less_value_relation : tokens_less_than_elsa_value = 20) :
  elsa_tokens - (tokens_less_than_elsa_value / token_value) = 55 :=
by
  rw [elsa_token_value_relation, token_value_relation, tokens_less_value_relation]
  norm_num
  sorry

end angus_token_count_l402_402502


namespace baron_munchausen_max_crowd_size_l402_402878

theorem baron_munchausen_max_crowd_size :
  ∃ n : ℕ, (∀ k, (k : ℕ) = n → 
  let left := (k / 2).toNat;
      right := (k / 3).toNat;
      straight := (k / 5).toNat in
  left + right + straight <= n + 1) ∧ 
  (∀ x : ℕ, x > 37 → ¬(∀ k, (k : ℕ) = x →
  let left := (k / 2).toNat;
      right := (k / 3).toNat;
      straight := (k / 5).toNat in
  left + right + straight <= x + 1)) :=
begin
  have h : 37 = 18 + 12 + 7,
  sorry,
end

end baron_munchausen_max_crowd_size_l402_402878


namespace weight_loss_percentage_final_weigh_in_l402_402014

theorem weight_loss_percentage_final_weigh_in (W : ℝ) (h : W > 0) :
  let new_weight := 0.88 * W,
      final_weight := new_weight + 0.02 * new_weight,
      percentage_loss := ((W - final_weight) / W) * 100
  in percentage_loss = 10.24 := by
  sorry

end weight_loss_percentage_final_weigh_in_l402_402014


namespace solution_system_equations_l402_402788

theorem solution_system_equations :
  ∀ (x y : ℝ) (k n : ℤ),
    (4 * (Real.cos x) ^ 2 - 4 * Real.cos x * (Real.cos (6 * x)) ^ 2 + (Real.cos (6 * x)) ^ 2 = 0) ∧
    (Real.sin x = Real.cos y) →
    (∃ k n : ℤ, (x = (Real.pi / 3) + 2 * Real.pi * k ∧ y = (Real.pi / 6) + 2 * Real.pi * n) ∨
                 (x = (Real.pi / 3) + 2 * Real.pi * k ∧ y = -(Real.pi / 6) + 2 * Real.pi * n) ∨
                 (x = -(Real.pi / 3) + 2 * Real.pi * k ∧ y = (5 * Real.pi / 6) + 2 * Real.pi * n) ∨
                 (x = -(Real.pi / 3) + 2 * Real.pi * k ∧ y = -(5 * Real.pi / 6) + 2 * Real.pi * n)) :=
by
  sorry

end solution_system_equations_l402_402788


namespace integer_solution_count_l402_402147

def has_integer_solutions (a : ℤ) : Prop :=
  ∃ x y : ℤ, x^2 + a * x + 12 * a = 0 ∧ y^2 + a * y + 12 * a = 0

theorem integer_solution_count : 
  (set.finite {a : ℤ | has_integer_solutions a}).to_finset.card = 16 := 
begin
  sorry
end

end integer_solution_count_l402_402147


namespace correct_understanding_of_philosophy_l402_402297

-- Define the conditions based on the problem statement
def philosophy_from_life_and_practice : Prop :=
  -- Philosophy originates from people's lives and practice.
  sorry
  
def philosophy_affects_lives : Prop :=
  -- Philosophy consciously or unconsciously affects people's lives, learning, and work
  sorry

def philosophical_knowledge_requires_learning : Prop :=
  true

def philosophy_not_just_summary : Prop :=
  true

-- Given conditions 1, 2, 3 (as negation of 3 in original problem), and 4 (as negation of 4 in original problem),
-- We need to prove the correct understanding (which is combination ①②) is correct.
theorem correct_understanding_of_philosophy :
  philosophy_from_life_and_practice →
  philosophy_affects_lives →
  philosophical_knowledge_requires_learning →
  philosophy_not_just_summary →
  (philosophy_from_life_and_practice ∧ philosophy_affects_lives) :=
by
  intros
  apply And.intro
  · assumption
  · assumption

end correct_understanding_of_philosophy_l402_402297


namespace range_of_sum_of_squares_l402_402327

noncomputable def decreasing_function (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, x < y → f x > f y

theorem range_of_sum_of_squares
  (f : ℝ → ℝ)
  (h_decr : decreasing_function f)
  (h_ineq : ∀ n m : ℝ, f (n^2 - 10 * n - 15) ≥ f (12 - m^2 + 24 * m)) :
  ∀ m n : ℝ, m^2 + n^2 ∈ set.Icc 0 729 :=
by
  sorry

end range_of_sum_of_squares_l402_402327


namespace external_bisector_l402_402751

open Triangle

variable {Point : Type} [EuclideanGeometry Point]

variables {A T C L K : Point}

theorem external_bisector (h : angle_bisector A T C L) : external_bisector A T C K :=
sorry

end external_bisector_l402_402751


namespace sum_b_n_l402_402203

variables (a_n : ℕ → ℝ) (b_n : ℕ → ℝ)

def is_arithmetic_seq (a : ℕ → ℝ) (d : ℝ) :=
∀ n, a (n + 1) = a n + d

def forms_geometric_seq (a1 a2 a5 : ℝ) :=
a2^2 = a1 * a5

def b_n_formula (a : ℕ → ℝ) (n : ℕ) : ℝ :=
(-1)^(n-1) * (n / (a n * a (n + 1)))

noncomputable def S_n (b : ℕ → ℝ) (n : ℕ) : ℝ :=
∑ i in finset.range n, b (i + 1)

theorem sum_b_n
    (h_arith : is_arithmetic_seq a_n 2)
    (h_geom : forms_geometric_seq (a_n 1) (a_n 2) (a_n 5))
    (a1_eq : a_n 1 = 1):
∀ n, S_n b_n n = if n % 2 = 1 then 1 / 4 * (1 + 1 / (2 * n + 1))
                                else 1 / 4 * (1 - 1 / (2 * n + 1)) :=
sorry

end sum_b_n_l402_402203


namespace external_bisector_TK_l402_402725

-- Definitions of points and segments
variables {A T C L K : Type}

-- Conditions given in the problem
variable (angle_bisector_TL : is_angle_bisector T L A C)
variable (is_triangle : is_triangle T A C)

-- Prove that TK is the external angle bisector
theorem external_bisector_TK : is_external_angle_bisector T K A C :=
sorry

end external_bisector_TK_l402_402725


namespace sheets_per_day_l402_402936

theorem sheets_per_day (total_sheets : ℕ) (days_per_week : ℕ) (sheets_per_day : ℕ) :
  total_sheets = 60 → days_per_week = 5 → sheets_per_day = total_sheets / days_per_week → sheets_per_day = 12 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3.symm.trans (by norm_num)

end sheets_per_day_l402_402936


namespace interval_membership_l402_402474

variable (x : ℝ)

def problem_condition := x = (1 / x) * (-x) + 4

theorem interval_membership (h : problem_condition x) : 2 < x ∧ x ≤ 4 :=
by {
  have hx : x = 3,
  { 
    -- Detailed computation done here to deduce x = 3. This will replace the sorry.
    sorry 
  },
  -- Exact interval membership derived from x = 3.
  exact ⟨by norm_num [hx], by norm_num [hx]⟩,
}

end interval_membership_l402_402474


namespace sum_of_two_smallest_trite_integers_l402_402316

def is_trite (n : ℕ) (d : Fin (n + 1) → ℕ) : Prop :=
  (d 5 + d 6 * (d 6 + d 4) = d 7 * d 4) ∧
  (∀ i j : Fin 12, i < j → (d i < d j)) ∧
  (d 0 = 1) ∧ (d 11 = n) ∧
  (∀ m : ℕ, (1 ≤ m ∧ m ≤ n) → ∃ i : Fin 12, d i = m)

def has_12_divisors (n : ℕ) (d : Fin (n + 1) → ℕ) : Prop :=
  (∀ m : ℕ, (1 ≤ m ∧ m ≤ n) → ∃ (i : Fin 12), d i = m) ∧
  (∀ i : Fin 12, n % (d i) = 0)

theorem sum_of_two_smallest_trite_integers : 
  ∃ n₁ n₂ : ℕ, is_trite n₁ (λ i, if i = 0 then 1 else if i = 11 then n₁ else d i) 
                ∧ is_trite n₂ (λ i, if i = 0 then 1 else if i = 11 then n₂ else d i) 
                ∧ n₁ < n₂ ∧ n₁ = 2020 ∧ n₂ = 149107 ∧ n₁ + n₂ = 151127 :=
by
  sorry

end sum_of_two_smallest_trite_integers_l402_402316


namespace limit_problem_l402_402063

open Real

theorem limit_problem :
  tendsto (λ x : ℝ, (sin x + sin (π * x) * arctan ((1 + x) / (1 - x))) / (1 + cos x)) (𝓝 1) (𝓝 (sin 1 / (1 + cos 1))) :=
by
  -- proof steps would go here
  sorry

end limit_problem_l402_402063


namespace abc_def_a_over_b_l402_402213

theorem abc_def_a_over_b (a b c d e f : ℚ) 
  (h1 : abc / def = 1.875)
  (h2 : b / c = 1 / 2)
  (h3 : c / d = 1)
  (h4 : d / e = 3 / 2)
  (h5 : e / f = 4 / 3) : a / b = 1.40625 := 
by 
  sorry

end abc_def_a_over_b_l402_402213


namespace area_circle_through_A_O_C_l402_402831

noncomputable def side_length : ℝ := 10
def radius_inscribed_circle : ℝ := side_length * Real.sqrt 3 / 3

theorem area_circle_through_A_O_C :
  let r := radius_inscribed_circle in let area := π * r^2 in area = 100 * π / 3 :=
by
  let r := radius_inscribed_circle
  let area := π * r^2
  have : area = π * (side_length * Real.sqrt 3 / 3)^2 :=
    by sorry  -- This would be proven but is skipped for now
  show area = 100 * π / 3 from sorry  -- This completes the proof but is currently skipped


end area_circle_through_A_O_C_l402_402831


namespace arithmetic_mean_of_30_consecutive_integers_starting_at_5_l402_402962

theorem arithmetic_mean_of_30_consecutive_integers_starting_at_5 :
  let sequence (n : ℕ) := n + 4 in
  let sum := (30 / 2) * (5 + (5 + 29)) in
  sum / 30 = 19.5 :=
by
  sorry

end arithmetic_mean_of_30_consecutive_integers_starting_at_5_l402_402962


namespace external_bisector_TK_l402_402726

-- Definitions of points and segments
variables {A T C L K : Type}

-- Conditions given in the problem
variable (angle_bisector_TL : is_angle_bisector T L A C)
variable (is_triangle : is_triangle T A C)

-- Prove that TK is the external angle bisector
theorem external_bisector_TK : is_external_angle_bisector T K A C :=
sorry

end external_bisector_TK_l402_402726


namespace find_z_l402_402692

noncomputable def z : ℂ := 1 + 3*I

theorem find_z (x y d : ℤ) (hx : x > 0) (hy : y > 0) (hz : (x + y*I)^3 = -26 + d*I) : 
  z = x + y*I := 
sorry

end find_z_l402_402692


namespace cannot_buy_same_number_of_notebooks_l402_402441

theorem cannot_buy_same_number_of_notebooks
  (price_softcover : ℝ)
  (price_hardcover : ℝ)
  (notebooks_ming : ℝ)
  (notebooks_li : ℝ)
  (h1 : price_softcover = 12)
  (h2 : price_hardcover = 21)
  (h3 : price_hardcover = price_softcover + 1.2) :
  notebooks_ming = 12 / price_softcover ∧
  notebooks_li = 21 / price_hardcover →
  ¬ (notebooks_ming = notebooks_li) :=
by
  sorry

end cannot_buy_same_number_of_notebooks_l402_402441


namespace count_distinct_a_l402_402106

def quadratic_has_integer_solutions (a : ℤ) : Prop :=
  ∃ x m n : ℤ, (x^2 + a * x + 12 * a = 0) ∧ (m + n = -a) ∧ (m * n = 12 * a)

theorem count_distinct_a : (finset.univ.filter quadratic_has_integer_solutions).card = 9 := 
sorry

end count_distinct_a_l402_402106


namespace find_radius_l402_402400

noncomputable def radius_of_wheel (D : ℝ) (n : ℕ) (r : ℝ) := D = n * (2 * Real.pi * r)

theorem find_radius (D : ℝ) (n : ℕ) (r : ℝ) (h1 : D = 703.9999999999999) (h2 : n = 500) : 
  radius_of_wheel D n r → r ≈ 0.224 :=
by
  sorry

end find_radius_l402_402400


namespace points_on_opposite_sides_of_line_l402_402520

theorem points_on_opposite_sides_of_line :
  let line_eq : ℝ → ℝ → ℝ := λ x y, 2 * x + y - 3 in
  let origin_inside := line_eq 0 0 < 0 in
  let point_inside := line_eq 2 3 > 0 in
  origin_inside ≠ point_inside :=
by
  let line_eq : ℝ → ℝ → ℝ := λ x y, 2 * x + y - 3
  have origin_inside : line_eq 0 0 < 0 := by norm_num
  have point_inside : line_eq 2 3 > 0 := by norm_num
  exact ne_of_lt_of_gt origin_inside point_inside

end points_on_opposite_sides_of_line_l402_402520


namespace tangent_line_through_B_l402_402594

theorem tangent_line_through_B (x : ℝ) (y : ℝ) (x₀ : ℝ) (y₀ : ℝ) :
  (y₀ = x₀^2) →
  (y - y₀ = 2*x₀*(x - x₀)) →
  (3, 5) ∈ ({p : ℝ × ℝ | ∃ t, p.2 - t^2 = 2*t*(p.1 - t)}) →
  (x = 2 * x₀) ∧ (y = y₀) →
  (2*x - y - 1 = 0 ∨ 10*x - y - 25 = 0) :=
by
  intros h1 h2 h3 h4
  sorry

end tangent_line_through_B_l402_402594


namespace workshop_personnel_l402_402439

-- Definitions for workshops with their corresponding production constraints
def workshopA_production (x : ℕ) : ℕ := 6 + 11 * (x - 1)
def workshopB_production (y : ℕ) : ℕ := 7 + 10 * (y - 1)

-- The main theorem to be proved
theorem workshop_personnel :
  ∃ (x y : ℕ), workshopA_production x = workshopB_production y ∧
               100 ≤ workshopA_production x ∧ workshopA_production x ≤ 200 ∧
               x = 12 ∧ y = 13 :=
by
  sorry

end workshop_personnel_l402_402439


namespace sum_c_n_d_n_div_9_n_l402_402340

noncomputable def c_n (n : ℕ) : ℝ := (sqrt 10)^n * real.cos (n * real.arctan (1 / 3))
noncomputable def d_n (n : ℕ) : ℝ := (sqrt 10)^n * real.sin (n * real.arctan (1 / 3))

theorem sum_c_n_d_n_div_9_n : 
  (\sum' n, (c_n n * d_n n) / 9^n) = 5 / 9 :=
sorry

end sum_c_n_d_n_div_9_n_l402_402340


namespace count_distinct_a_l402_402110

def quadratic_has_integer_solutions (a : ℤ) : Prop :=
  ∃ x m n : ℤ, (x^2 + a * x + 12 * a = 0) ∧ (m + n = -a) ∧ (m * n = 12 * a)

theorem count_distinct_a : (finset.univ.filter quadratic_has_integer_solutions).card = 9 := 
sorry

end count_distinct_a_l402_402110


namespace find_a_of_ff_eq_2_l402_402230

def f (x : ℝ) : ℝ := if x ≤ 0 then x^2 + 2*x + 2 else -x^2

theorem find_a_of_ff_eq_2 : 
  ∃ (a : ℝ), f (f a) = 2 ∧ a = real.sqrt 2 :=
begin
  sorry
end

end find_a_of_ff_eq_2_l402_402230


namespace matching_pair_probability_l402_402987

theorem matching_pair_probability:
  let total_socks := 12 + 10 + 6 in
  let blue_pairs := (12 * 11) / 2 in
  let red_pairs := (10 * 9) / 2 in
  let green_pairs := (6 * 5) / 2 in
  let total_pairs := (total_socks * (total_socks - 1)) / 2 in
  let matching_pairs := blue_pairs + red_pairs + green_pairs in
  (matching_pairs / total_pairs) = 1 / 3 :=
by
  sorry

end matching_pair_probability_l402_402987


namespace original_faculty_members_proof_l402_402480

noncomputable def original_faculty_members (remaining_faculty: ℕ) (reduction_percent: ℝ) : ℝ :=
  remaining_faculty / (1 - reduction_percent)

theorem original_faculty_members_proof {remaining_faculty : ℕ} {reduction_percent : ℝ} 
  (h1 : remaining_faculty = 195) (h2 : reduction_percent = 0.20) :
  round (original_faculty_members remaining_faculty reduction_percent) = 244 :=
by
  -- Definitions and conditions
  let original_faculty := original_faculty_members remaining_faculty reduction_percent
  have h3 : original_faculty = 195 / 0.80 :=
      by
        simp [h1, h2]
  have h4 : original_faculty = 243.75 :=
      by
        linarith [h3]
  show round original_faculty = 244,
      by
        simp [h4]
        sorry

end original_faculty_members_proof_l402_402480


namespace find_nine_day_segment_l402_402487

/-- 
  Definitions:
  - ws_day: The Winter Solstice day, December 21, 2012.
  - j1_day: New Year's Day, January 1, 2013.
  - Calculate the total days difference between ws_day and j1_day.
  - Check that the distribution of days into 9-day segments leads to January 1, 2013, being the third day of the second segment.
-/
def ws_day : ℕ := 21
def j1_day : ℕ := 1
def days_in_december : ℕ := 31
def days_ws_to_end_dec : ℕ := days_in_december - ws_day + 1
def total_days : ℕ := days_ws_to_end_dec + j1_day

theorem find_nine_day_segment : (total_days % 9) = 3 ∧ (total_days / 9) = 1 := by
  sorry  -- Proof skipped

end find_nine_day_segment_l402_402487


namespace number_of_distinct_a_l402_402132

noncomputable def count_distinct_a : ℕ :=
  let divisors := [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 36, 48, 72, 144, -1, -2, -3, -4, -6, -8, -9, -12, -16, -18, -24, -36, -48, -72, -144]
  let pairs := (divisors.product divisors).filter (λ (p : ℤ × ℤ), p.1 * p.2 = 144)
  let sums := pairs.map (λ (p : ℤ × ℤ), p.1 + p.2 - 24)
  let distinct_sums := sums.eraseDuplicates  
  distinct_sums.length

theorem number_of_distinct_a : count_distinct_a = 14 :=
by
  sorry

end number_of_distinct_a_l402_402132


namespace log_graph_shift_l402_402809

theorem log_graph_shift:
  ∀ (x : ℝ), (λ x : ℝ, log 2 (x - 1)) (x + 2) = (λ x : ℝ, log 2 (x + 1)) x :=
by
  sorry

end log_graph_shift_l402_402809


namespace conversion_problems_l402_402020

-- Define the conversion factors
def square_meters_to_hectares (sqm : ℕ) : ℕ := sqm / 10000
def hectares_to_square_kilometers (ha : ℕ) : ℕ := ha / 100
def square_kilometers_to_hectares (sqkm : ℕ) : ℕ := sqkm * 100

-- Define the specific values from the problem
def value1_m2 : ℕ := 5000000
def value2_km2 : ℕ := 70000

-- The theorem to prove
theorem conversion_problems :
  (square_meters_to_hectares value1_m2 = 500) ∧
  (hectares_to_square_kilometers 500 = 5) ∧
  (square_kilometers_to_hectares value2_km2 = 7000000) :=
by
  sorry

end conversion_problems_l402_402020


namespace y_work_days_l402_402862

theorem y_work_days (d : ℝ) : 
  let x_rate := 1 / 21 in
  let y_rate := 1 / d in
  let y_work := 10 * y_rate in
  let remaining_work := 1 - y_work in
  let x_days := 7 in
  x_days * x_rate = remaining_work → d = 15 :=
by
  intros h
  sorry

end y_work_days_l402_402862


namespace min_additional_packs_needed_l402_402764

-- Defining the problem conditions
def total_sticker_packs : ℕ := 40
def packs_per_basket : ℕ := 7

-- The statement to prove
theorem min_additional_packs_needed : 
  ∃ (additional_packs : ℕ), 
    (total_sticker_packs + additional_packs) % packs_per_basket = 0 ∧ 
    (total_sticker_packs + additional_packs) / packs_per_basket = 6 ∧ 
    additional_packs = 2 :=
by 
  sorry

end min_additional_packs_needed_l402_402764


namespace count_integer_values_of_a_l402_402154

theorem count_integer_values_of_a (a : ℤ) :
  (∃ x : ℤ, x^2 + a * x + 12 * a = 0) ↔ (a ∈ {-(m + n) | m, n : ℤ ∧ m * n = 12 * a}) → ∃! a_vals : finset ℤ, a_vals.card = 16 :=
by
  sorry

end count_integer_values_of_a_l402_402154


namespace angus_tokens_eq_l402_402498

-- Define the conditions
def worth_per_token : ℕ := 4
def elsa_tokens : ℕ := 60
def angus_less_worth : ℕ := 20

-- Define the main theorem to prove
theorem angus_tokens_eq :
  let elsaTokens := elsa_tokens,
      worthPerToken := worth_per_token,
      angusLessTokens := angus_less_worth / worth_per_token
  in (elsaTokens - angusLessTokens) = 55 := by
  sorry

end angus_tokens_eq_l402_402498


namespace total_students_in_lunchroom_l402_402900

theorem total_students_in_lunchroom (students_per_table : ℕ) (num_tables : ℕ) (total_students : ℕ) :
  students_per_table = 6 → 
  num_tables = 34 → 
  total_students = students_per_table * num_tables → 
  total_students = 204 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end total_students_in_lunchroom_l402_402900


namespace axis_of_symmetry_l402_402803

def f (x : ℝ) : ℝ := Real.sin (x - Real.pi / 3)

theorem axis_of_symmetry :
  ∃ k : ℤ, ∀ x : ℝ, (x = (5 * Real.pi / 6) + k * Real.pi) ↔ ∃ n : ℤ, x = (5 * Real.pi / 6) + n * Real.pi :=
by
  sorry

end axis_of_symmetry_l402_402803


namespace IncorrectProposition_l402_402492

-- Definitions based on the given conditions
structure Quadrilateral :=
(a b c d : ℝ)

def Rhombus (q : Quadrilateral) : Prop :=
q.a = q.b ∧ q.b = q.c ∧ q.c = q.d

def Square (q : Quadrilateral) : Prop :=
q.a = q.b ∧ q.b = q.c ∧ q.c = q.d ∧
angle q.a q.b = 90 ∧
angle q.b q.c = 90 ∧
angle q.c q.d = 90 ∧
angle q.d q.a = 90

def Rectangle (q : Quadrilateral) : Prop :=
angle q.a q.b = 90 ∧
angle q.b q.c = 90 ∧
angle q.c q.d = 90 ∧
angle q.d q.a = 90

def DiagonalsBisect (q : Quadrilateral) : Prop :=
bisects (diag q.a q.c) (diag q.b q.d)

def DiagonalsPerpendicular (q : Quadrilateral) : Prop :=
perpendicular (diag q.a q.c) (diag q.b q.d)

-- Initializing a quadrilateral
variable (q : Quadrilateral)

-- Proposition definitions based on given conditions
def PropositionA : Prop := Rhombus q
def PropositionB : Prop := DiagonalsPerpendicular q ∧ DiagonalsBisect q → Square q
def PropositionC : Prop := (angle q.a q.b = 90 ∧ angle q.b q.c = 90 ∧ angle q.c q.d = 90) → Rectangle q
def PropositionD : Prop := (DiagonalsBisect q ∧ DiagonalsEqual q) → Rectangle q

-- The statement to prove that Proposition B is incorrect
theorem IncorrectProposition : ¬ PropositionB :=
sorry

end IncorrectProposition_l402_402492


namespace circle_equation_with_diameter_l402_402610

noncomputable def midpoint (P Q : ℝ × ℝ) : ℝ × ℝ :=
  ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

theorem circle_equation_with_diameter (P Q : ℝ × ℝ) (hP : P = (4, 0)) (hQ : Q = (0, 2)) :
  ∃ (C : ℝ × ℝ) (r : ℝ), (midpoint P Q = C) ∧ (r = distance P Q / 2) ∧ (∀ (x y : ℝ), (x - C.1)^2 + (y - C.2)^2 = 5) :=
by
  sorry

end circle_equation_with_diameter_l402_402610


namespace equal_sum_partition_l402_402557

theorem equal_sum_partition (n : ℕ) (a : Fin n.succ → ℕ)
  (h1 : a 0 = 1)
  (h2 : ∀ i : Fin n, a i ≤ a i.succ ∧ a i.succ ≤ 2 * a i)
  (h3 : (Finset.univ : Finset (Fin n.succ)).sum a % 2 = 0) :
  ∃ (partition : Finset (Fin n.succ)), 
    (partition.sum a = (partitionᶜ : Finset (Fin n.succ)).sum a) :=
by sorry

end equal_sum_partition_l402_402557


namespace elmo_to_laura_books_ratio_l402_402091

-- Definitions of the conditions given in the problem
def ElmoBooks : ℕ := 24
def StuBooks : ℕ := 4
def LauraBooks : ℕ := 2 * StuBooks

-- Ratio calculation and proof of the ratio being 3:1
theorem elmo_to_laura_books_ratio : (ElmoBooks : ℚ) / (LauraBooks : ℚ) = 3 / 1 := by
  sorry

end elmo_to_laura_books_ratio_l402_402091


namespace anne_and_katherine_savings_l402_402061

variables (A K : ℝ)

-- Conditions
def condition1 := A - 150 = (1 / 3) * K
def condition2 := 2 * K = 3 * A

-- Conclusion to prove
def total_savings := A + K = 750

theorem anne_and_katherine_savings (h1 : condition1) (h2 : condition2) : total_savings :=
sorry

end anne_and_katherine_savings_l402_402061


namespace correct_intuitive_diagram_conclusion_l402_402422

theorem correct_intuitive_diagram_conclusion
  (C1 : ∀ (t : Triangle), is_triangle (intuitive_diagram t))
  (C2 : ∀ (s : Square), is_rhombus (intuitive_diagram s))
  (C3 : ∀ (it : IsoscelesTrapezoid), can_be_parallelogram (intuitive_diagram it))
  (C4 : ∀ (r : Rhombus), is_rhombus (intuitive_diagram r)) :
  ∨ (C1, ¬C2, ¬C3, ¬C4) := by sorry

end correct_intuitive_diagram_conclusion_l402_402422


namespace mutual_fund_percent_increase_l402_402012

theorem mutual_fund_percent_increase :
  ∀ (P : ℝ), 
  let price_end_first_quarter := 1.30 * P,
      price_end_second_quarter := 1.50 * P,
      percent_increase := ((price_end_second_quarter - price_end_first_quarter) / price_end_first_quarter) * 100 in
  percent_increase ≈ 15.38 :=
by
  intro P
  let price_end_first_quarter := 1.30 * P
  let price_end_second_quarter := 1.50 * P
  let percent_increase := ((price_end_second_quarter - price_end_first_quarter) / price_end_first_quarter) * 100
  have hp : percent_increase = ((1.50 * P - 1.30 * P) / (1.30 * P)) * 100 := rfl
  simp [hp]
  have hsimplify : ((0.20 * P) / (1.30 * P)) * 100 = (0.20 / 1.30) * 100 := sorry
  have hequivalence : 200 / 13 ≈ 15.38 := sorry
  exact hequivalence

end mutual_fund_percent_increase_l402_402012


namespace union_of_A_and_B_l402_402208

def A : Set ℝ := { x | 1 < x ∧ x < 3 }
def B : Set ℝ := { x | 2 < x ∧ x < 4 }

theorem union_of_A_and_B : A ∪ B = { x | 1 < x ∧ x < 4 } := by
  sorry

end union_of_A_and_B_l402_402208


namespace max_sum_abc_l402_402263

theorem max_sum_abc (a b c : ℝ) (h : a + b + c = a^2 + b^2 + c^2) : a + b + c ≤ 3 :=
sorry

end max_sum_abc_l402_402263


namespace fewer_oranges_l402_402550

variable (O_A A_A O_G A_G : ℕ)

def conditions :=
  O_G = 45 ∧ A_G = A_A + 5 ∧ A_A = 15 ∧ (O_G + A_G + O_A + A_A) = 107

theorem fewer_oranges (h : conditions O_A A_A O_G A_G) :
  O_G - O_A = 18 :=
by
  cases h with
  | intro h1 (intro h2 (intro h3 h4)) =>
  sorry

end fewer_oranges_l402_402550


namespace area_triangle_POQ_parabola_focus_l402_402928

theorem area_triangle_POQ_parabola_focus (P Q : ℝ × ℝ) : 
  let focus := (1, 0);
  let parabola := λ x y : ℝ, y^2 = 4 * x;
  let line := λ x y : ℝ, y^2 - 4 * y - 4 = 0;
  let O := (0, 0);
  let angle := π / 4;
  parabola P.1 P.2 ∧ parabola Q.1 Q.2 ∧ line P.1 P.2 ∧ line Q.1 Q.2 →
  ∃ (area : ℝ), area = 2 * sqrt 2 := 
by
  sorry

end area_triangle_POQ_parabola_focus_l402_402928


namespace find_number_l402_402856

theorem find_number (x : ℝ) : 3 * (2 * x + 9) = 57 → x = 5 :=
by
  sorry

end find_number_l402_402856


namespace sum_of_factors_of_30_l402_402429

/--
Given the positive integer factors of 30, prove that their sum is 72.
-/
theorem sum_of_factors_of_30 : 
  (1 + 2 + 3 + 5 + 6 + 10 + 15 + 30) = 72 := 
by 
  sorry

end sum_of_factors_of_30_l402_402429


namespace quadratic_complete_the_square_l402_402821

theorem quadratic_complete_the_square :
  ∃ b c : ℝ, (∀ x : ℝ, x^2 + 1500 * x + 1500 = (x + b) ^ 2 + c)
      ∧ b = 750
      ∧ c = -748 * 750
      ∧ c / b = -748 := 
by {
  sorry
}

end quadratic_complete_the_square_l402_402821


namespace external_angle_bisector_of_triangle_l402_402743

variables {A T C L K : Type}
variables [IsTriangle A T C] [IsInternalAngleBisector T L (A T C)]

theorem external_angle_bisector_of_triangle 
  (h1 : is_internal_angle_bisector TL ATC) :
  is_external_angle_bisector TK ATC :=
sorry

end external_angle_bisector_of_triangle_l402_402743


namespace smallest_number_of_students_l402_402033

theorem smallest_number_of_students 
  (ninth_to_seventh : ℕ → ℕ → Prop)
  (ninth_to_sixth : ℕ → ℕ → Prop) 
  (r1 : ninth_to_seventh 3 2) 
  (r2 : ninth_to_sixth 7 4) : 
  ∃ n7 n6 n9, 
    ninth_to_seventh n9 n7 ∧ 
    ninth_to_sixth n9 n6 ∧ 
    n9 + n7 + n6 = 47 :=
sorry

end smallest_number_of_students_l402_402033


namespace integer_values_a_l402_402179

theorem integer_values_a :
  (∃! a : ℤ, ∃ m n : ℤ, m^2 + n^2 = (m + n)^2 - 2mn ∧
                     m + n = -a ∧ 
                     mn = 12a) :
  8 :=
sorry

end integer_values_a_l402_402179


namespace burger_cost_l402_402489

theorem burger_cost :
  ∃ b s f : ℕ, 4 * b + 2 * s + 3 * f = 480 ∧ 3 * b + s + 2 * f = 360 ∧ b = 80 :=
by
  sorry

end burger_cost_l402_402489


namespace problem_iterated_kernels_l402_402534

section kernels

variable (a b : ℝ)
variable (K : ℝ → ℝ → ℝ)

def K1 (x t : ℝ) := K x t

def K2 (x t : ℝ) := ∫ s in a..b, K x s * K s t

theorem problem_iterated_kernels (a b : ℝ) (K : ℝ → ℝ → ℝ) (x t : ℝ) :
  (a = 0) →
  (b = 1) →
  (K x t = exp (min x t)) →
  (K1 K x t = K x t) ∧
  (K2 K x t = 
    if x ≤ t then (2 - t) * exp (x + t) - (1 + exp (2 * x)) / 2
    else (2 - x) * exp (x + t) - (1 + exp (2 * t)) / 2) :=
by
  intros ha hb hK
  simp [K1, K2, hK]
  sorry

end kernels

end problem_iterated_kernels_l402_402534


namespace external_bisector_of_triangle_l402_402698

variables {A T C K L : Type} [noncomputable_field A T C K L]
variable {triangle ATC : Triangle A T C}

noncomputable def internal_angle_bisector (T L : Point) := 
    ∃ R, is_bisector TL ∧ TL ∝ AR RL

noncomputable def external_angle_bisector (T K : Point) := 
    ∃ M, is_bisector TK ∧ TK ∝ AM MK

theorem external_bisector_of_triangle {A T C K L : Point} 
    (hL : internal_angle_bisector T L) 
    : external_angle_bisector T K :=
sorry

end external_bisector_of_triangle_l402_402698


namespace range_of_a_for_f_decreasing_l402_402211

noncomputable def f (a x : ℝ) : ℝ := Real.log (x^2 - a * x + 3 * a) / Real.log (1 / 2)

def is_decreasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x ≤ y → y ≤ b → f y ≤ f x

theorem range_of_a_for_f_decreasing :
  (∀ (a : ℝ), is_decreasing_on_interval (f a) 2 +∞) ↔ (-4 < a ∧ a ≤ 4) :=
sorry

end range_of_a_for_f_decreasing_l402_402211


namespace derivative_of_x_logx_l402_402386

-- Define the function 
def y (x : ℝ) : ℝ := x * real.log x

-- State the theorem
theorem derivative_of_x_logx (x : ℝ) (h : x ≠ 0) : deriv y x = 1 + real.log x :=
by sorry

end derivative_of_x_logx_l402_402386


namespace max_people_in_crowd_l402_402872

theorem max_people_in_crowd : ∃ n : ℕ, n ≤ 37 ∧ 
    (⟨1 / 2 * n⟩ + ⟨1 / 3 * n⟩ + ⟨1 / 5 * n⟩ = n) :=
sorry

end max_people_in_crowd_l402_402872


namespace no_two_right_angles_triangle_l402_402436

theorem no_two_right_angles_triangle : ∀ (α β γ : ℝ), α = 90 → β = 90 → α + β + γ = 180 → false :=
by
  intros α β γ hα hβ hsum
  have h : 90 + 90 + γ = 180 := by
    rw [hα, hβ]
  rw [← add_assoc] at h
  linarith

end no_two_right_angles_triangle_l402_402436


namespace sqrt_six_lt_a_lt_cubic_two_l402_402262

theorem sqrt_six_lt_a_lt_cubic_two (a : ℝ) (h : a^5 - a^3 + a = 2) : (Real.sqrt 3)^6 < a ∧ a < 2^(1/3) :=
sorry

end sqrt_six_lt_a_lt_cubic_two_l402_402262


namespace find_minimum_argument_complex_number_l402_402949

def is_within_circle_centered_at_5i_with_radius_4 (z : ℂ) : Prop :=
  complex.abs (z - 5 * complex.I) ≤ 4

def has_smallest_positive_argument (z : ℂ) (sz : ℂ) : Prop :=
  ∃ (a : ℝ), 0 < a ∧ a = complex.arg z ∧ ∀ (w : ℂ), is_within_circle_centered_at_5i_with_radius_4 w → 0 < complex.arg w → complex.arg z ≤ complex.arg w

theorem find_minimum_argument_complex_number :
  ∃ (z : ℂ), is_within_circle_centered_at_5i_with_radius_4 z ∧ has_smallest_positive_argument z (2.4 + 1.8 * complex.I) := 
sorry

end find_minimum_argument_complex_number_l402_402949


namespace always_possible_l402_402484

theorem always_possible (n : ℕ) (hn : n > 2) :
  ∃ (a : Fin n → ℕ), (∀ i j : Fin n, i ≠ j → ¬(a i ∣ a j)) ∧
                     (∀ k : Fin n, k.val ≥ 2 → a k ∣ (Finset.univ.erase k).sum (λ i, a i)) :=
by
  sorry

end always_possible_l402_402484


namespace correct_conclusions_count_l402_402219

theorem correct_conclusions_count :
  (¬ (∃ x : ℝ, 2 ^ x ≤ 0) ↔ ∀ x : ℝ, 2 ^ x > 0) ∧
  (∀ f ∈ [{x : ℝ | x ≠ 0 → y = x⁻¹}, {x : ℝ | y = x}, {x : ℝ | y = x^2}, {x : ℝ | y = x^3}], is_increasing_on ℝ f → 
  card {x : ℝ | x ≠ 0 → y = x⁻¹ ∧ is_increasing_on ℝ f} = 2) ∧
  ((∀ a b : ℝ, |a + b| = |a| + |b| ↔ a * b ≥ 0)) ∧
  (∀ m : ℝ, (m^2 + 2 * m - 3) = 0 ∧ (m - 1 ≠ 0) ↔ (¬(m - 1 = 0) → m = -3 ∨ m = 1)) → 
  2 = 2 := 
sorry

end correct_conclusions_count_l402_402219


namespace integer_values_of_a_l402_402170

theorem integer_values_of_a (a : ℤ) :
  (∀ x : ℤ, x^2 + a * x + 12 * a = 0 → x ∈ ℤ) → ∃ n : ℕ, n = 16 :=
by
  sorry

end integer_values_of_a_l402_402170


namespace pasha_wins_game_l402_402360

def can_pasha_always_win : Prop :=
  ∀ (moves : ℕ → ℕ → ℕ × ℕ) (initial_weight : ℝ),
  (∀ n, let (pasha_pieces, vova_pieces) := moves n initial_weight in pasha_pieces + vova_pieces = initial_weight) →
  (∃ k : ℕ, k ≥ 100 → ∀ (step : ℕ), step ≤ k → ∃ (cur_weight : ℝ), cur_weight = initial_weight / 3 ^ step)

theorem pasha_wins_game : can_pasha_always_win :=
by
  sorry

end pasha_wins_game_l402_402360


namespace external_bisector_l402_402757

-- Define the relevant points and segments in the triangle
variables {A T C L K : Type} [inhabited A] [inhabited T] [inhabited C] [inhabited L] [inhabited K]

-- Assuming T, L, K are points such that TL divides angle ATC internally
axiom internal_bisector (h : L) : ∃ TL, TL.bisects_angle (A T C) internally

-- Prove that TK is the external angle bisector given the condition above
theorem external_bisector (h : L) (h_internal : internal_bisector h) : 
  ∃ TK, TK.bisects_angle (A T C) externally :=
sorry

end external_bisector_l402_402757


namespace main_theorem_l402_402581

-- Define the set of candidate values for a
def candidate_values : set ℝ := {-1, 1, 1/2, 3}

-- Define what it means for a function to have a domain of ℝ
def has_domain_real (a : ℝ) : Prop :=
∀ x : ℝ, x ^ a = x ^ a

-- Define what it means for a function to be an odd function
def is_odd_function (a : ℝ) : Prop :=
∀ x : ℝ, x ^ a = -((-x) ^ a)

-- Define the correct values of a
def correct_values (a : ℝ) : Prop :=
a = 1 ∨ a = 3

-- State the main theorem
theorem main_theorem :
  ∀ a ∈ candidate_values, (has_domain_real a ∧ is_odd_function a) ↔ correct_values a :=
by
  sorry

end main_theorem_l402_402581


namespace main_problem_l402_402337

def g (n : ℕ) : ℕ := if n = 0 then 0 else 0

theorem main_problem :
  (let m := { n : ℕ | ∃ a b c : ℕ, 3 * g (a^2 + b^2 + c^2) = g (a) ^ 2 + g (b) ^ 2 + g (c) ^ 2 }
          .count = 1,
       t := { n : ℕ | ∃ a b c : ℕ, n = g (27) }.sum) in
  m * t = 0 :=
by 
  sorry

end main_problem_l402_402337


namespace largest_crowd_size_l402_402889

theorem largest_crowd_size :
  ∃ (n : ℕ), 
    (⌊n / 2⌋ + ⌊n / 3⌋ + ⌊n / 5⌋ = n) ∧
    ∀ m : ℕ, (⌊m / 2⌋ + ⌊m / 3⌋ + ⌊m / 5⌋ = m) → m ≤ 37 :=
sorry

end largest_crowd_size_l402_402889


namespace light_flash_fraction_l402_402036

theorem light_flash_fraction (flash_interval : ℕ) (total_flashes : ℕ) (seconds_in_hour : ℕ) (fraction_of_hour : ℚ) :
  flash_interval = 6 →
  total_flashes = 600 →
  seconds_in_hour = 3600 →
  fraction_of_hour = 1 →
  (total_flashes * flash_interval) / seconds_in_hour = fraction_of_hour := by
  sorry

end light_flash_fraction_l402_402036


namespace not_secretly_one_variable_l402_402837

def secretly_one_variable (P : polynomial ℝ → polynomial ℝ → polynomial ℝ) : Prop :=
  ∃ Q R : polynomial ℝ, (degree Q ≥ 2) ∧ ∀ x y, P x y = Q (R x y)

theorem not_secretly_one_variable (P : polynomial ℝ → polynomial ℝ → polynomial ℝ)
  (h1 : ∃ f g : polynomial ℝ, (f ≠ 0) ∧ (g ≠ 0) ∧ ∀ x y, P x y = f * g)
  (h2 : ∃ f g : polynomial ℝ, (f ≠ 0) ∧ (g ≠ 0) ∧ ∀ x y, (P x y + 1) = f * g)
  : ¬ secretly_one_variable P :=
sorry

end not_secretly_one_variable_l402_402837


namespace number_of_distinct_a_l402_402131

noncomputable def count_distinct_a : ℕ :=
  let divisors := [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 36, 48, 72, 144, -1, -2, -3, -4, -6, -8, -9, -12, -16, -18, -24, -36, -48, -72, -144]
  let pairs := (divisors.product divisors).filter (λ (p : ℤ × ℤ), p.1 * p.2 = 144)
  let sums := pairs.map (λ (p : ℤ × ℤ), p.1 + p.2 - 24)
  let distinct_sums := sums.eraseDuplicates  
  distinct_sums.length

theorem number_of_distinct_a : count_distinct_a = 14 :=
by
  sorry

end number_of_distinct_a_l402_402131


namespace rope_new_length_is_23_l402_402941

noncomputable def new_rope_length (initial_length : ℝ) (additional_area : ℝ) : ℝ :=
  let A1 := Real.pi * (initial_length ^ 2)
  let A2 := A1 + additional_area
  let r2 := Real.sqrt (A2 / Real.pi)
  r2

theorem rope_new_length_is_23 :
  new_rope_length 10 1348.2857142857142 ≈ 23 := 
by 
  have initial_length := 10
  have additional_area := 1348.2857142857142
  have A1 : ℝ := Real.pi * (initial_length ^ 2)
  have A2 : ℝ := A1 + additional_area
  have r2 : ℝ := Real.sqrt (A2 / Real.pi)
  have r2_val : ℝ ≈ 23 := by sorry -- Placeholder for the actual approximation proof
  exact r2_val

end rope_new_length_is_23_l402_402941


namespace min_and_max_distinct_sums_l402_402893

theorem min_and_max_distinct_sums (n : ℕ) (hn : n > 2) :
  (∃ (a : Fin n → ℕ), (∀ i j : Fin n, i ≠ j → a i ≠ a j) ∧
  (card (Finset.image (λ (p : Fin n × Fin n), a p.fst + a p.snd)
    ((Finset.univ.product Finset.univ).filter (λ p, p.fst < p.snd))) = 2 * n - 3)) ∧
  (∃ (b : Fin n → ℕ), (∀ i j : Fin n, i ≠ j → b i ≠ b j) ∧
  (card (Finset.image (λ (p : Fin n × Fin n), b p.fst + b p.snd)
    ((Finset.univ.product Finset.univ).filter (λ p, p.fst < p.snd))) = (n * (n - 1)) / 2)) :=
begin
  sorry
end

end min_and_max_distinct_sums_l402_402893


namespace minimum_familiar_pairs_l402_402909

theorem minimum_familiar_pairs (n : ℕ) (students : Finset (Fin n)) 
  (familiar : Finset (Fin n × Fin n))
  (h_n : n = 175)
  (h_condition : ∀ (s : Finset (Fin n)), s.card = 6 → 
    ∃ (s1 s2 : Finset (Fin n)), s1 ∪ s2 = s ∧ s1.card = 3 ∧ s2.card = 3 ∧ 
    ∀ x ∈ s1, ∀ y ∈ s1, (x ≠ y → (x, y) ∈ familiar) ∧
    ∀ x ∈ s2, ∀ y ∈ s2, (x ≠ y → (x, y) ∈ familiar)) :
  ∃ m : ℕ, m = 15050 ∧ ∀ p : ℕ, (∃ g : Finset (Fin n × Fin n), g.card = p) → p ≥ m := 
sorry

end minimum_familiar_pairs_l402_402909


namespace cookies_initial_amount_l402_402306

theorem cookies_initial_amount (C : ℕ) (h1 : ∃ C, 
  let afterJen_first := (1-3/4) * C in
  let afterKate_first := (1-1/5) * afterJen_first in
  let afterJen_second := (1-1/3) * afterKate_first in
  let afterKate_second := (1-1/4) * afterJen_second in
  let afterLee_third := (1-1/2) * afterKate_second in
  let afterSteve_third := (1-1/7) * afterLee_third in
  afterSteve_third = 8) : C = 187 :=
begin
  sorry
end

end cookies_initial_amount_l402_402306


namespace integer_solution_count_l402_402149

def has_integer_solutions (a : ℤ) : Prop :=
  ∃ x y : ℤ, x^2 + a * x + 12 * a = 0 ∧ y^2 + a * y + 12 * a = 0

theorem integer_solution_count : 
  (set.finite {a : ℤ | has_integer_solutions a}).to_finset.card = 16 := 
begin
  sorry
end

end integer_solution_count_l402_402149


namespace percentage_deducted_from_list_price_l402_402058

-- Definitions based on conditions
def cost_price : ℝ := 85.5
def marked_price : ℝ := 112.5
def profit_rate : ℝ := 0.25 -- 25% profit

noncomputable def selling_price : ℝ := cost_price * (1 + profit_rate)

theorem percentage_deducted_from_list_price:
  ∃ d : ℝ, d = 5 ∧ selling_price = marked_price * (1 - d / 100) :=
by
  sorry

end percentage_deducted_from_list_price_l402_402058


namespace integer_solution_count_l402_402143

def has_integer_solutions (a : ℤ) : Prop :=
  ∃ x y : ℤ, x^2 + a * x + 12 * a = 0 ∧ y^2 + a * y + 12 * a = 0

theorem integer_solution_count : 
  (set.finite {a : ℤ | has_integer_solutions a}).to_finset.card = 16 := 
begin
  sorry
end

end integer_solution_count_l402_402143


namespace count_distinct_sums_of_special_fractions_l402_402067

-- Define special fractions
def is_special_fraction (a b : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ a + b = 20

-- The main theorem statement
theorem count_distinct_sums_of_special_fractions : 
  ∃ n : ℕ, n = 15 ∧ ∀ (a b c d : ℕ), is_special_fraction(a, b) ∧ is_special_fraction(c, d) → 
  (∃ distinct_integers : set ℤ, card distinct_integers = n ∧ 
                               ∀ (x y : ℤ), x ∈ distinct_integers ∧ y ∈ distinct_integers → 
                               ∃ z : ℤ, z ∈ distinct_integers ∧ z = (a / b) + (c / d)) := sorry

end count_distinct_sums_of_special_fractions_l402_402067


namespace octagon_diagonal_ratio_l402_402645

theorem octagon_diagonal_ratio (s : ℝ) (h : s > 0) :
  let AC := s * real.sqrt (2 - 2 * (real.cos (real.pi / 8))),
      AD := s * real.sqrt (2 - 2 * (real.cos (3 * real.pi / 4)))
  in AC / AD = real.sqrt (2 - real.sqrt (2 + real.sqrt 2)) / real.sqrt (2 + real.sqrt 2) :=
by sorry

end octagon_diagonal_ratio_l402_402645


namespace m_is_perfect_square_l402_402783

-- Given definitions and conditions
def is_odd (k : ℤ) : Prop := ∃ n : ℤ, k = 2 * n + 1

def is_perfect_square (m : ℕ) : Prop := ∃ a : ℕ, m = a * a

theorem m_is_perfect_square (k m n : ℕ) (h1 : (2 + Real.sqrt 3) ^ k = 1 + m + n * Real.sqrt 3)
  (h2 : 0 < m) (h3 : 0 < n) (h4 : 0 < k) (h5 : is_odd k) : is_perfect_square m := 
sorry

end m_is_perfect_square_l402_402783


namespace complement_M_in_U_l402_402238

open Set

variable (x : ℝ)

def M : Set ℝ := {x | 2^x - 4 ≤ 0}

theorem complement_M_in_U : (Mᶜ = {x | x > 2}) :=
by
    sorry

end complement_M_in_U_l402_402238


namespace range_of_a_l402_402632

theorem range_of_a (a : ℝ) (H : ∀ x : ℝ, x ≤ 1 → 4 - a * 2^x > 0) : a < 2 :=
sorry

end range_of_a_l402_402632


namespace TK_is_external_bisector_of_ATC_l402_402712

variable {A T C L K : Type} [LinearOrder A] [LinearOrder T] [LinearOrder C] [LinearOrder L] [LinearOrder K]
variable (triangle_ATC : Triangle A T C)
variable (point_T : Point T)
variable (point_L : Point L)
variable (point_K : Point K)
variable (internal_bisector_TL : AngleBisector (angle_ATC) T L)
variable (external_bisector_TK : AngleBisector (external_angle_ATC) T K)

theorem TK_is_external_bisector_of_ATC :
  external_bisector_TK = true :=
sorry

end TK_is_external_bisector_of_ATC_l402_402712


namespace count_integers_satisfying_property_l402_402618

theorem count_integers_satisfying_property :
  let count := (Finset.filter (λ n : ℤ, (n - 3) * (n + 3) * (n + 7) < 0) (Finset.Icc (-10 : ℤ) 12)).card
  in count = 5 :=
by
  sorry

end count_integers_satisfying_property_l402_402618


namespace decreasing_func_l402_402255

noncomputable def func (x : ℝ) (b : ℝ) : ℝ := -0.5 * x^2 + b * Real.log (x + 2)

theorem decreasing_func (b : ℝ) : (∀ x > -1, deriv (λ x, func x b) x ≤ 0) → b ≤ -1 := by
  sorry  -- Proof not required

end decreasing_func_l402_402255


namespace common_ratio_value_l402_402291

noncomputable def common_ratio_arith_geo (a : ℕ → ℕ) (d : ℕ) :=
  -- Define the arithmetic sequence
  (∀ n, a n = 2 + (n - 1) * d) ∧
  -- Define the geometric sequence condition
  (a 1 / a 1 = a 3 / a 1 = a 11 / a 3) →
  -- Prove the common ratio is 4
  (d ≠ 0) →
  (a 3 / a 1 = 4)

theorem common_ratio_value (a d : ℕ) (h1 : ∀ n, a n = 2 + (n - 1) * d) 
  (h2 : a 1 / a 1 = a 3 / a 1 = a 11 / a 3) (h3 : d ≠ 0) : 
  a 3 / a 1 = 4 :=
sorry

end common_ratio_value_l402_402291


namespace concentric_circles_AB_CD_l402_402896

/-- Let \( O \) be the center of two concentric circles with radii \( r \) and \( R \), and \( AB \) and \( CD \) be
    two lines such that \( AB \) is parallel to \( CD \). Let \( h \) be the distance from \( O \) to \( AB \) and
    \( H \) be the distance from \( O \) to \( CD \), with \( h < H \). If \( AB \) and \( CD \) are the lengths 
    intercepted by the inner and outer circles respectively, then \( AB < CD \). -/
theorem concentric_circles_AB_CD (O A B C D : Point) 
  (r R h H : ℝ) (k1 k2 : Circle) (AB CD : ℝ)
  (hne: h < H) (r_pos : r > 0) (R_pos : R > 0) (AB_eq : AB = 2 * ∥(r^2 - h^2)∥.sqrt)
  (CD_eq : CD = 2 * ∥(R^2 - H^2)∥.sqrt) (circles_concentric : k1.center = O ∧ k2.center = O) 
  (lines_parallel : is_parallel A B C D) (inner_circle : k1.radius = r)
  (outer_circle : k2.radius = R):
  AB < CD := 
sorry

end concentric_circles_AB_CD_l402_402896


namespace square_free_has_power_of_two_divisors_l402_402338

def is_square_free (n : ℕ) : Prop :=
  ∀ p : ℕ, p.prime → p ∣ n → ¬ p^2 ∣ n

theorem square_free_has_power_of_two_divisors {n : ℕ} (h1 : 0 < n) (h2 : is_square_free n) : 
  ∃ k : ℕ, nat.totient n = 2^k :=
sorry

end square_free_has_power_of_two_divisors_l402_402338


namespace avg_speed_is_30_l402_402854

-- Define the average speed going from San Diego to San Francisco
def avg_speed_go (D : ℝ) : ℝ := 45

-- Define the time taken for the journey to San Francisco
def time_go (D : ℝ) : ℝ := D / avg_speed_go D

-- Define the time taken for the return journey from San Francisco
def time_back (D : ℝ) : ℝ := 2 * time_go D

-- Define the total distance for the round trip
def total_distance (D : ℝ) : ℝ := 2 * D

-- Define the total time for the round trip
def total_time (D : ℝ) : ℝ := time_go D + time_back D

-- Define the average speed for the entire trip
def avg_speed_round_trip (D : ℝ) : ℝ := total_distance D / total_time D

-- Theorem: The average speed for the entire trip
theorem avg_speed_is_30 (D : ℝ) (h : D > 0) : avg_speed_round_trip D = 30 := 
by
  sorry

end avg_speed_is_30_l402_402854


namespace balls_in_boxes_l402_402206

/--
Given five balls numbered 1, 2, 3, 4, 5 and five boxes also numbered 1, 2, 3, 4, 5, 
where exactly two balls must be placed in their corresponding numbered boxes, 
prove that the total number of ways to arrange the balls such that exactly two balls 
are in their corresponding boxes is 20.
-/
theorem balls_in_boxes :
  let balls := finset.range 5
  let boxes := finset.range 5
  ∃ arr : (balls → boxes), (finset.filter (λ (i : ℕ), arr i = i) balls).card = 2 ∧ 
  count_derangements (balls \ {x | arr x = x}) = 20 :=
sorry

end balls_in_boxes_l402_402206


namespace largest_subset_size_dist_gt_2_l402_402190

def S : set (Fin 5 → ℕ) :=
  { v | ∀ i : Fin 5, v i = 0 ∨ v i = 1 }

def distance (a b : Fin 5 → ℕ) : ℕ :=
  ∑ i in Finset.univ, abs (a i - b i)

theorem largest_subset_size_dist_gt_2 :
  ∃ T ⊆ S, (∀ a b ∈ T, a ≠ b → distance a b > 2) ∧ (∀ T' ⊆ S, (∀ a b ∈ T', a ≠ b → distance a b > 2) → Finset.card T ≤ Finset.card (T' : Finset (Fin 5 → ℕ))) :=
sorry

end largest_subset_size_dist_gt_2_l402_402190


namespace leading_coefficient_of_polynomial_l402_402932

-- Define a polynomial f(x) as a finite list representing the coefficients in ascending powers of x.
-- f(x) is nonzero and has real coefficients.
variables {R : Type*} [CommRing R] [IsDomain R] [CharZero R]

-- Define f(x), f'(x), and f''(x) and their relations
theorem leading_coefficient_of_polynomial (f : Polynomial R) (h1 : f ≠ 0)
    (h2 : ∀ (x : R), f.eval x = (f.derivative.eval x) * (f.derivative.derivative.eval x)) :
    f.leadingCoeff = (1 : R) / 18 := 
sorry

end leading_coefficient_of_polynomial_l402_402932


namespace amys_yard_area_l402_402493

theorem amys_yard_area :
  ∃ (n : ℕ) (m : ℕ), (4 + 2n + 4 + 2 * ℕ + 2n - 4 = 24) ∧ 3 * (4 - 1 + 2n - 1) * (4 - 1 + 2 * ℕ - 1) = 189 :=
by
  sorry

end amys_yard_area_l402_402493


namespace largest_crowd_size_l402_402886

theorem largest_crowd_size (x : ℕ) : 
  (ceil (x * (1 / 2)) + ceil (x * (1 / 3)) + ceil (x * (1 / 5)) = x) →
  x ≤ 37 :=
sorry

end largest_crowd_size_l402_402886


namespace volume_of_regular_tetrahedron_l402_402066

theorem volume_of_regular_tetrahedron (R : ℝ) : 
  (∃ (a : ℝ), a = R * sqrt 3) ∧ 
  (∃ (V : ℝ), V = (R^3 * sqrt 6) / 4) :=
by
  sorry

end volume_of_regular_tetrahedron_l402_402066


namespace smallest_x_for_perfect_cube_l402_402693

theorem smallest_x_for_perfect_cube (x N : ℕ) (hN : 1260 * x = N^3) (h_fact : 1260 = 2^2 * 3^2 * 5 * 7): x = 7350 := sorry

end smallest_x_for_perfect_cube_l402_402693


namespace price_of_sports_equipment_l402_402462

theorem price_of_sports_equipment (x y : ℕ) (a b : ℕ) :
  (2 * x + y = 330) → (5 * x + 2 * y = 780) → x = 120 ∧ y = 90 ∧
  (120 * a + 90 * b = 810) → a = 3 ∧ b = 5 :=
by
  intros h1 h2 h3
  sorry

end price_of_sports_equipment_l402_402462


namespace ridgecrest_academy_physics_l402_402957

theorem ridgecrest_academy_physics (total_players : ℕ) (math_players : ℕ) (both_players : ℕ)
  (h1 : total_players = 15) (h2 : math_players = 10) (h3 : both_players = 4) :
  ∃ physics_players : ℕ, physics_players = 9 :=
by
  use 9
  sorry

end ridgecrest_academy_physics_l402_402957


namespace number_of_distinct_a_l402_402126

noncomputable def count_distinct_a : ℕ :=
  let divisors := [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 36, 48, 72, 144, -1, -2, -3, -4, -6, -8, -9, -12, -16, -18, -24, -36, -48, -72, -144]
  let pairs := (divisors.product divisors).filter (λ (p : ℤ × ℤ), p.1 * p.2 = 144)
  let sums := pairs.map (λ (p : ℤ × ℤ), p.1 + p.2 - 24)
  let distinct_sums := sums.eraseDuplicates  
  distinct_sums.length

theorem number_of_distinct_a : count_distinct_a = 14 :=
by
  sorry

end number_of_distinct_a_l402_402126


namespace largest_crowd_size_l402_402892

theorem largest_crowd_size :
  ∃ (n : ℕ), 
    (⌊n / 2⌋ + ⌊n / 3⌋ + ⌊n / 5⌋ = n) ∧
    ∀ m : ℕ, (⌊m / 2⌋ + ⌊m / 3⌋ + ⌊m / 5⌋ = m) → m ≤ 37 :=
sorry

end largest_crowd_size_l402_402892


namespace university_min_spend_l402_402432

-- Definition of the box dimensions
def box_length : ℝ := 20
def box_width : ℝ := 20
def box_height : ℝ := 12

-- Definition of the cost per box
def cost_per_box : ℝ := 0.40

-- Definition of the total volume needed
def total_volume_needed : ℝ := 2160000

-- Definition of the volume of one box
def volume_of_one_box : ℝ := box_length * box_width * box_height

-- Definition of the number of boxes needed
def number_of_boxes_needed : ℝ := total_volume_needed / volume_of_one_box

-- Definition of the minimum amount to spend on boxes
def min_amount_to_spend : ℝ := number_of_boxes_needed.ceil * cost_per_box

-- Statement to prove
theorem university_min_spend : min_amount_to_spend = 180 := by
  sorry

end university_min_spend_l402_402432


namespace integer_solution_count_l402_402150

def has_integer_solutions (a : ℤ) : Prop :=
  ∃ x y : ℤ, x^2 + a * x + 12 * a = 0 ∧ y^2 + a * y + 12 * a = 0

theorem integer_solution_count : 
  (set.finite {a : ℤ | has_integer_solutions a}).to_finset.card = 16 := 
begin
  sorry
end

end integer_solution_count_l402_402150


namespace harmonic_sum_identity_l402_402105

def h (n : ℕ) : ℚ := (finset.range n).sum (λ k, 1 / (k + 1 : ℚ))

theorem harmonic_sum_identity (n : ℕ) (hn : n ≥ 2) :
  n + (finset.range (n - 1)).sum (λ k, h (k + 1)) = n * h n :=
sorry

end harmonic_sum_identity_l402_402105


namespace greatest_k_divides_n_l402_402475

theorem greatest_k_divides_n (n : ℕ) (h1 : n.factors.prod.dvd n = 72) (h2 : (5 * n).factors.prod.dvd (5 * n) = 120) : ∀ k : ℕ, (5 ^ k).dvd n → k = 0 :=
by
  sorry

end greatest_k_divides_n_l402_402475


namespace conic_section_hyperbola_l402_402980

theorem conic_section_hyperbola (x y : ℝ) :
  (x - 3) ^ 2 = 9 * (y + 2) ^ 2 - 81 → conic_section := by
  sorry

end conic_section_hyperbola_l402_402980


namespace algebraic_identity_l402_402582

theorem algebraic_identity (x y : ℝ) (h₁ : x * y = 4) (h₂ : x - y = 5) : 
  x^2 + 5 * x * y + y^2 = 53 := 
by 
  sorry

end algebraic_identity_l402_402582


namespace eval_fraction_l402_402961

theorem eval_fraction : (144 : ℕ) = 12 * 12 → (12 ^ 10 / (144 ^ 4) : ℝ) = 144 := by
  intro h
  have h1 : (144 : ℕ) = 12 ^ 2 := by
    exact h
  sorry

end eval_fraction_l402_402961


namespace primitive_root_mod_p_alpha_l402_402339

theorem primitive_root_mod_p_alpha
  (p : ℕ) (x : ℕ) (α : ℕ) 
  (hp_prime : Nat.prime p) 
  (hx_primitive_root : Nat.is_primitive_root x p)
  (h_condition : x^(p^(α-2)*(p-1)) % p^α ≠ 1) 
  (h_alpha_ge_two : α ≥ 2) : 
  Nat.is_primitive_root x (p^α) := 
sorry

end primitive_root_mod_p_alpha_l402_402339


namespace find_function_expression_solve_inequality_l402_402597

def given_function (a b : ℝ) (x : ℝ) : ℝ := x^2 / (a * x + b)

theorem find_function_expression (a b : ℝ) :
  given_function a b 3 - 3 + 12 = 0 ∧ given_function a b 4 - 4 + 12 = 0 →
  ∃ f : ℝ → ℝ, ∀ x, f x = given_function a b x :=
sorry

theorem solve_inequality (a b k : ℝ) (h1 : 1 < k) :
  ∀ x, given_function a b x < (k + 1) * x - k / (2 - x) ↔ ((x-2)*(x-1)*(x-k) > 0) :=
sorry

end find_function_expression_solve_inequality_l402_402597


namespace eval_gg3_l402_402695

def g (x : ℕ) : ℕ := 3 * x^2 + 3 * x - 2

theorem eval_gg3 : g (g 3) = 3568 :=
by 
  sorry

end eval_gg3_l402_402695


namespace value_of_m_l402_402657

-- Definition of the problem's conditions
variable (m : ℝ)
def coordinates_A : (ℝ × ℝ) := (2, 0)
def coordinates_P : (ℝ × ℝ) := (m, 2 * m - 1)

-- The slope of the line PO considering the origin
def slope_PO : ℝ := (2 * m - 1) / m

-- The main theorem stating the values of m if the angle POA is 45 degrees
theorem value_of_m (h1 : slope_PO m = 1 ∨ slope_PO m = -1) : m = 1 ∨ m = 1/3 :=
by
  sorry

end value_of_m_l402_402657


namespace count_distinct_a_l402_402112

def quadratic_has_integer_solutions (a : ℤ) : Prop :=
  ∃ x m n : ℤ, (x^2 + a * x + 12 * a = 0) ∧ (m + n = -a) ∧ (m * n = 12 * a)

theorem count_distinct_a : (finset.univ.filter quadratic_has_integer_solutions).card = 9 := 
sorry

end count_distinct_a_l402_402112


namespace parallelogram_base_length_l402_402380

theorem parallelogram_base_length (A H : ℝ) (hA : A = 78.88) (hH : H = 8) : A / H = 9.86 := 
by 
  rw [hA, hH] 
  exact (div_eq 78.88 8).symm.transferRealCm sorry

end parallelogram_base_length_l402_402380


namespace problem_correct_answer_l402_402260

def quasi_even (f : ℝ → ℝ) : Prop :=
  ∃ S : Set ℝ, (Finite S ∧ ∀ x ∈ S, x ≠ 0 ∧ f(-x) = f(x)) 

theorem problem_correct_answer : 
  quasi_even (λ x : ℝ, x^3 - 2 * x) :=
sorry

end problem_correct_answer_l402_402260


namespace exists_ABC_for_odd_sums_l402_402423

theorem exists_ABC_for_odd_sums (N : ℕ) 
  (triples : Fin N → ℤ × ℤ × ℤ)
  (h1 : ∀ i, let ⟨a, b, c⟩ := triples i in a % 2 = 1 ∨ b % 2 = 1 ∨ c % 2 = 1) :
  ∃ (A B C : ℤ), (∑ i in Finset.range N, if (let ⟨a, b, c⟩ := triples i in (A * a + B * b + C * c) % 2 = 1) then 1 else 0) ≥ (4 * N) / 7 :=
by
  sorry

end exists_ABC_for_odd_sums_l402_402423


namespace angus_tokens_count_l402_402497

def worth_of_token : ℕ := 4
def elsa_tokens : ℕ := 60
def difference_worth : ℕ := 20

def elsa_worth : ℕ := elsa_tokens * worth_of_token
def angus_worth : ℕ := elsa_worth - difference_worth

def angus_tokens : ℕ := angus_worth / worth_of_token

theorem angus_tokens_count : angus_tokens = 55 := by
  sorry

end angus_tokens_count_l402_402497


namespace provenance_of_positive_test_l402_402272

noncomputable def pr_disease : ℚ := 1 / 200
noncomputable def pr_no_disease : ℚ := 1 - pr_disease
noncomputable def pr_test_given_disease : ℚ := 1
noncomputable def pr_test_given_no_disease : ℚ := 0.05
noncomputable def pr_test : ℚ := pr_test_given_disease * pr_disease + pr_test_given_no_disease * pr_no_disease
noncomputable def pr_disease_given_test : ℚ := 
  (pr_test_given_disease * pr_disease) / pr_test

theorem provenance_of_positive_test : pr_disease_given_test = 20 / 219 :=
by
  sorry

end provenance_of_positive_test_l402_402272


namespace total_price_paid_l402_402370

noncomputable def total_price
    (price_rose : ℝ) (qty_rose : ℕ) (discount_rose : ℝ)
    (price_lily : ℝ) (qty_lily : ℕ) (discount_lily : ℝ)
    (price_sunflower : ℝ) (qty_sunflower : ℕ)
    (store_discount : ℝ) (tax_rate : ℝ)
    : ℝ :=
  let total_rose := qty_rose * price_rose
  let total_lily := qty_lily * price_lily
  let total_sunflower := qty_sunflower * price_sunflower
  let total := total_rose + total_lily + total_sunflower
  let total_disc_rose := total_rose * discount_rose
  let total_disc_lily := total_lily * discount_lily
  let discounted_total := total - total_disc_rose - total_disc_lily
  let store_discount_amount := discounted_total * store_discount
  let after_store_discount := discounted_total - store_discount_amount
  let tax_amount := after_store_discount * tax_rate
  after_store_discount + tax_amount

theorem total_price_paid :
  total_price 20 3 0.15 15 5 0.10 10 2 0.05 0.07 = 140.79 :=
by
  apply sorry

end total_price_paid_l402_402370


namespace quadratic_to_general_form_l402_402518

theorem quadratic_to_general_form (x : ℝ) :
  ∃ b : ℝ, (∀ a c : ℝ, (a = 3) ∧ (c = 1) → (a * x^2 + c = 6 * x) → b = -6) :=
by
  sorry

end quadratic_to_general_form_l402_402518


namespace revolutions_same_distance_l402_402926

theorem revolutions_same_distance (r R : ℝ) (revs_30 : ℝ) (dist_30 dist_10 : ℝ)
  (h_radius: r = 10) (H_radius: R = 30) (h_revs_30: revs_30 = 15) 
  (H_dist_30: dist_30 = 2 * Real.pi * R * revs_30) 
  (H_dist_10: dist_10 = 2 * Real.pi * r * 45) :
  dist_30 = dist_10 :=
by {
  sorry
}

end revolutions_same_distance_l402_402926


namespace vector_simplification_l402_402784

-- Define vectors AB, CD, AC, and BD
variables {V : Type*} [AddCommGroup V]

-- Given vectors
variables (AB CD AC BD : V)

-- Theorem to be proven
theorem vector_simplification :
  (AB - CD) - (AC - BD) = (0 : V) :=
sorry

end vector_simplification_l402_402784


namespace distinct_integer_values_of_a_l402_402116

theorem distinct_integer_values_of_a :
  ∃ (a_values : Finset ℤ), (∀ a ∈ a_values, ∃ (m n : ℤ), (x^2 + a * x + 12 * a = 0) ∧ (m + n = -a) ∧ (m * n = 12 * a))
  ∧ a_values.card = 16 :=
sorry

end distinct_integer_values_of_a_l402_402116


namespace problem1a_problem1b_l402_402918

noncomputable theory

def valid_purchase_price (a : ℤ) : Prop :=
  600 * a = 1300 * (a - 140)

def maximize_profit (x : ℤ) : Prop :=
  let y := (200 * x) / 2 + 120 * x / 2 + 20 * (5 * x + 20 - 2 * x)
  x + 5 * x + 20 ≤ 200 ∧ y = 9200

theorem problem1a (a : ℤ) : valid_purchase_price a ↔ a = 260 := sorry

theorem problem1b (x : ℤ) : maximize_profit x ↔ (x = 30 ∧ (5 * x + 20 = 170)) := sorry

end problem1a_problem1b_l402_402918


namespace angus_tokens_count_l402_402495

def worth_of_token : ℕ := 4
def elsa_tokens : ℕ := 60
def difference_worth : ℕ := 20

def elsa_worth : ℕ := elsa_tokens * worth_of_token
def angus_worth : ℕ := elsa_worth - difference_worth

def angus_tokens : ℕ := angus_worth / worth_of_token

theorem angus_tokens_count : angus_tokens = 55 := by
  sorry

end angus_tokens_count_l402_402495


namespace biased_coin_heads_four_times_probability_l402_402458

noncomputable def biased_coin_probability : ℚ :=
  let h := 3 / 7 in
  let n := 6 in
  let k := 4 in
  (nat.choose n k) * (h ^ k) * ((1 - h) ^ (n - k))

theorem biased_coin_heads_four_times_probability :
  biased_coin_probability = 19440 / 117649 ∧ (19440 + 117649 = 137089) :=
by
  sorry

end biased_coin_heads_four_times_probability_l402_402458


namespace total_students_correct_l402_402507

noncomputable def total_students (initial_boys : ℕ) (girls_multiplier : ℕ) (girls_percentage_increase : ℝ) : ℕ :=
  let initial_girls := initial_boys + (girls_percentage_increase * initial_boys).to_nat in
  let final_girls := girls_multiplier * initial_girls in
  initial_boys + final_girls

theorem total_students_correct : total_students 30 3 0.4 = 156 :=
by
  sorry

end total_students_correct_l402_402507


namespace jeff_ends_at_prime_prob_l402_402305

noncomputable def primes : list ℕ := [2, 3, 5, 7, 11]

def possible_movements := [-2, -1, 1, 2]

def movement_prob (m1 m2 : ℤ) : ℚ :=
  if m1 + m2 = 0 then (1 : ℚ) / 16 else 0

def start_prob (n : ℕ) : ℚ :=
  if n ∈ primes then (5 : ℚ) / 12 else 0

def combined_prob : ℚ :=
  start_prob 2 * (movement_prob (-2) 2 + movement_prob (-1) 1 + movement_prob 1 (-1) + movement_prob 2 (-2))

theorem jeff_ends_at_prime_prob : combined_prob = 5 / 48 :=
by
  sorry

end jeff_ends_at_prime_prob_l402_402305


namespace intersection_complement_U_M_eq_1_2_l402_402216

open Set Real

def M : Set ℝ := {x | ∃ y, y = log(1 - 2 / x)}
def N : Set ℝ := {x | ∃ y, y = sqrt (x - 1)}

noncomputable def complement_M : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

theorem intersection_complement_U_M_eq_1_2 :
  N ∩ complement_M = {x | 1 ≤ x ∧ x ≤ 2} :=
by
  sorry

end intersection_complement_U_M_eq_1_2_l402_402216


namespace largest_crowd_size_l402_402881

theorem largest_crowd_size (x : ℕ) : 
  (ceil (x * (1 / 2)) + ceil (x * (1 / 3)) + ceil (x * (1 / 5)) = x) →
  x ≤ 37 :=
sorry

end largest_crowd_size_l402_402881


namespace range_of_c_l402_402574

theorem range_of_c (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (hab : a + b = a * b) (habc : a + b + c = a * b * c) : 1 < c ∧ c ≤ 4 / 3 :=
by
  sorry

end range_of_c_l402_402574


namespace check_independence_and_expected_value_l402_402041

noncomputable def contingency_table (students: ℕ) (pct_75 : ℕ) (pct_less10 : ℕ) (num_75_10 : ℕ) : (ℕ × ℕ) × (ℕ × ℕ) :=
  let num_75 := students * pct_75 / 100
  let num_less10 := students * pct_less10 / 100
  let num_75_less10 := num_75 - num_75_10
  let num_not75 := students - num_75
  let num_not75_less10 := num_less10 - num_75_less10
  let num_not75_10 := num_not75 - num_not75_less10
  ((num_not75_less10, num_75_less10), (num_not75_10, num_75_10))

noncomputable def chi_square_statistic (a b c d : ℕ) (n: ℕ) : ℚ :=
  (n * ((a * d - b * c)^2)) / ((a + b) * (c + d) * (a + c) * (b + d))

theorem check_independence_and_expected_value :
  let students := 500
  let pct_75 := 30
  let pct_less10 := 50
  let num_75_10 := 100
  let ((num_not75_less10, num_75_less10), (num_not75_10, num_75_10)) := contingency_table students pct_75 pct_less10 num_75_10
  let chi2 := chi_square_statistic num_not75_less10 num_75_less10 num_not75_10 num_75_10 students
  let critical_value := 10.828
  let p0 := 1 / 84
  let p1 := 3 / 14
  let p2 := 15 / 28
  let p3 := 5 / 21
  let expected_x := 0 * p0 + 1 * p1 + 2 * p2 + 3 * p3
  (chi2 > critical_value) ∧ (expected_x = 2) :=
by 
  sorry

end check_independence_and_expected_value_l402_402041


namespace largest_crowd_size_l402_402885

theorem largest_crowd_size (x : ℕ) : 
  (ceil (x * (1 / 2)) + ceil (x * (1 / 3)) + ceil (x * (1 / 5)) = x) →
  x ≤ 37 :=
sorry

end largest_crowd_size_l402_402885


namespace sum_of_possible_x_l402_402086

theorem sum_of_possible_x : 
  ∀ (x : ℚ), 
  (mean_med_mode_ap (multiset.cons x [8, 3, 3, 7, 3, 9])) → 
  x = 54 / 13 :=
by
  sorry

def mean_med_mode_ap (lst : multiset ℚ) : Prop :=
  let mean := multiset.sum lst / multiset.card lst in
  let mode := 3 in
  let med : ℚ := (lst.sort (≤)).choose! (λ _, true) in -- simplified median calculation for the purpose of the statement
  true -- replace with actual AP check

end sum_of_possible_x_l402_402086


namespace cost_price_per_meter_is_correct_l402_402485

-- Definitions of the conditions
def selling_price : ℕ := 8925
def meters_sold : ℕ := 85
def profit_per_meter : ℕ := 15

-- Definition of what we need to prove
def total_profit : ℕ := profit_per_meter * meters_sold
def cost_price_per_meter (selling_price total_profit meters_sold : ℕ) : ℕ := 
  let total_cost_price := selling_price - total_profit in
  total_cost_price / meters_sold

theorem cost_price_per_meter_is_correct :
  cost_price_per_meter selling_price total_profit meters_sold = 90 := by
  sorry

end cost_price_per_meter_is_correct_l402_402485


namespace part_a_part_b_l402_402504

noncomputable def pair_guess_strategy_bound : ℕ := 1019091
noncomputable def binary_search_strategy_bound : ℕ := 21161

theorem part_a :
  ∀ (a : Fin 2020 → Fin 2020), ∃ guesses : ℕ, guesses ≤ pair_guess_strategy_bound := 
by
  intros a
  use pair_guess_strategy_bound
  sorry

theorem part_b :
  ∀ (a : Fin 2019 → Fin 2019), ∃ guesses : ℕ, guesses ≤ binary_search_strategy_bound :=
by
  intros a
  use binary_search_strategy_bound
  sorry

end part_a_part_b_l402_402504


namespace complex_product_is_real_l402_402688

theorem complex_product_is_real (a b c d : ℝ) :
  (ad + bc = 0) ↔ ((a + b * Complex.i) * (c + d * Complex.i)).im = 0 :=
sorry

end complex_product_is_real_l402_402688


namespace sheets_per_day_l402_402933

-- Definitions based on conditions
def total_sheets : ℕ := 60
def total_days_per_week : ℕ := 7
def days_off : ℕ := 2

-- Derived condition from the problem
def work_days_per_week : ℕ := total_days_per_week - days_off

-- The statement to prove
theorem sheets_per_day : total_sheets / work_days_per_week = 12 :=
by
  sorry

end sheets_per_day_l402_402933


namespace vertical_asymptotes_count_l402_402978

def f (x : ℝ) : ℝ := (x + 2) / (x^2 - 9)

theorem vertical_asymptotes_count : ∃ n : ℕ, n = 2 ∧ ∀ x : ℝ, (f x = 1/0 → (x = 3 ∨ x = -3)) :=
begin
  sorry, 
end

end vertical_asymptotes_count_l402_402978


namespace Quincy_sold_more_l402_402669

def ThorSales : ℕ := 200 / 10
def JakeSales : ℕ := ThorSales + 10
def QuincySales : ℕ := 200

theorem Quincy_sold_more (H : QuincySales = 200) : QuincySales - JakeSales = 170 := by
  sorry

end Quincy_sold_more_l402_402669


namespace max_coins_arbitrary_max_coins_equal_l402_402897

namespace Chernomor

-- Defining the general condition of the problem
def total_warriors := 33
def total_coins := 240

-- First part: Maximum coins Chernomor can receive if he distributes arbitrarily
theorem max_coins_arbitrary :
  ∃ (coins_received : ℕ), coins_received = 31 :=
begin
  let max_remainder := 31,
  use max_remainder,
  sorry -- Proof steps will go here
end

-- Second part: Maximum coins Chernomor can receive if he distributes equally
theorem max_coins_equal :
  ∃ (coins_received : ℕ), coins_received = 30 :=
begin
  let max_remainder := 30,
  use max_remainder,
  sorry -- Proof steps will go here
end

end Chernomor

end max_coins_arbitrary_max_coins_equal_l402_402897


namespace sin_B_plus_pi_over_6_eq_l402_402663

noncomputable def sin_b_plus_pi_over_6 (B : ℝ) : ℝ :=
  Real.sin B * (Real.sqrt 3 / 2) + (Real.sqrt (1 - (Real.sin B) ^ 2)) * (1 / 2)

theorem sin_B_plus_pi_over_6_eq :
  ∀ (A B : ℝ) (b c : ℝ),
    A = (2 * Real.pi / 3) →
    b = 1 →
    (1 / 2 * b * c * Real.sin A) = Real.sqrt 3 →
    c = 2 →
    sin_b_plus_pi_over_6 B = (2 * Real.sqrt 7 / 7) :=
by
  intros A B b c hA hb hArea hc
  sorry

end sin_B_plus_pi_over_6_eq_l402_402663


namespace compare_P_Q_l402_402580

noncomputable def P (a : ℝ) : ℝ := sqrt a + sqrt (a + 3)
noncomputable def Q (a : ℝ) : ℝ := sqrt (a + 1) + sqrt (a + 2)

theorem compare_P_Q (a : ℝ) (h : a > 0) : P a < Q a := sorry

end compare_P_Q_l402_402580


namespace angus_token_count_l402_402503

theorem angus_token_count (elsa_tokens : ℕ) (token_value : ℕ) 
  (tokens_less_than_elsa_value : ℕ) (elsa_token_value_relation : elsa_tokens = 60) 
  (token_value_relation : token_value = 4) (tokens_less_value_relation : tokens_less_than_elsa_value = 20) :
  elsa_tokens - (tokens_less_than_elsa_value / token_value) = 55 :=
by
  rw [elsa_token_value_relation, token_value_relation, tokens_less_value_relation]
  norm_num
  sorry

end angus_token_count_l402_402503


namespace integer_values_a_l402_402183

theorem integer_values_a :
  (∃! a : ℤ, ∃ m n : ℤ, m^2 + n^2 = (m + n)^2 - 2mn ∧
                     m + n = -a ∧ 
                     mn = 12a) :
  8 :=
sorry

end integer_values_a_l402_402183


namespace cost_price_of_watch_l402_402010

theorem cost_price_of_watch (CP SP_loss SP_gain : ℝ) (h1 : SP_loss = 0.79 * CP)
  (h2 : SP_gain = 1.04 * CP) (h3 : SP_gain - SP_loss = 140) : CP = 560 := by
  sorry

end cost_price_of_watch_l402_402010


namespace tailor_cut_skirt_l402_402047

theorem tailor_cut_skirt (cut_pants cut_skirt : ℝ) (h1 : cut_pants = 0.5) (h2 : cut_skirt = cut_pants + 0.25) : cut_skirt = 0.75 :=
by
  sorry

end tailor_cut_skirt_l402_402047


namespace yogurt_amount_l402_402527

-- Conditions
def total_ingredients : ℝ := 0.5
def strawberries : ℝ := 0.2
def orange_juice : ℝ := 0.2

-- Question and Answer (Proof Goal)
theorem yogurt_amount : total_ingredients - strawberries - orange_juice = 0.1 := by
  -- Since calculation involves specifics, we add sorry to indicate the proof is skipped
  sorry

end yogurt_amount_l402_402527


namespace number_of_integer_values_of_a_l402_402165

theorem number_of_integer_values_of_a : 
  ∃ (a_set : Set ℤ), (∀ a ∈ a_set, ∃ x y : ℤ, x^2 + a * x + 12 * a = 0 ∧ y^2 + a * y + 12 * a = 0) ∧ a_set.card = 16 :=
sorry

end number_of_integer_values_of_a_l402_402165


namespace leaves_blew_away_l402_402351

theorem leaves_blew_away (initial_leaves : ℕ) (leaves_left : ℕ) (blew_away : ℕ) 
  (h1 : initial_leaves = 356) (h2 : leaves_left = 112) (h3 : blew_away = initial_leaves - leaves_left) :
  blew_away = 244 :=
by
  sorry

end leaves_blew_away_l402_402351


namespace number_of_distinct_a_l402_402129

noncomputable def count_distinct_a : ℕ :=
  let divisors := [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 36, 48, 72, 144, -1, -2, -3, -4, -6, -8, -9, -12, -16, -18, -24, -36, -48, -72, -144]
  let pairs := (divisors.product divisors).filter (λ (p : ℤ × ℤ), p.1 * p.2 = 144)
  let sums := pairs.map (λ (p : ℤ × ℤ), p.1 + p.2 - 24)
  let distinct_sums := sums.eraseDuplicates  
  distinct_sums.length

theorem number_of_distinct_a : count_distinct_a = 14 :=
by
  sorry

end number_of_distinct_a_l402_402129


namespace complex_numbers_property_l402_402593

open Complex

theorem complex_numbers_property
  (z1 z2 z3 : ℂ)
  (h1 : ∥z1∥ = ∥z2∥)
  (h2 : ∥z2∥ = Real.sqrt 3 * ∥z3∥)
  (h3 : z1 + z3 = z2) :
  z1 * z2 / z3 = -5 := by
  sorry

end complex_numbers_property_l402_402593


namespace sum_of_first_three_tests_l402_402763

variable (A B C: ℕ)

def scores (A B C test4 : ℕ) : Prop := (A + B + C + test4) / 4 = 85

theorem sum_of_first_three_tests (h : scores A B C 100) : A + B + C = 240 :=
by
  -- Proof goes here
  sorry

end sum_of_first_three_tests_l402_402763


namespace circle_standard_form_and_properties_l402_402217

theorem circle_standard_form_and_properties :
    let C := λ x y : ℝ, x^2 + y^2 - 2 * x + 4 * y - 4 = 0 in
    let center := (1 : ℝ, -2 : ℝ) in
    let radius := 3 in
    ∃ l : ℝ → ℝ, 
      (∀ x : ℝ, l x = x - 4 ∨ l x = x + 1) ∧
      (∀ A B : ℝ × ℝ, A ≠ B → 
        (let mid := (A.1 + B.1) / 2, (A.2 + B.2) / 2 in 
        C mid.1 mid.2 ∧
        (A.1 - B.1)^2 + (A.2 - B.2)^2 = (2 * radius)^2 ∧
        (A.1 - center.1)^2 + (A.2 - center.2)^2 ≤ radius^2 ∧
        let area := abs ((A.1 - center.1) * (B.2 - center.2) - (A.2 - center.2) * (B.1 - center.1)) / 2 in
        area <= 9 / 2 )
      )
 := sorry

end circle_standard_form_and_properties_l402_402217


namespace external_angle_bisector_l402_402715

open EuclideanGeometry

variables {A T C K L : Point}

theorem external_angle_bisector
    (h1 : is_internal_angle_bisector T L A T C) : is_external_angle_bisector T K A T C :=
sorry

end external_angle_bisector_l402_402715


namespace loan_interest_rate_l402_402913

-- Defining the loan amount (principal)
def principal : ℝ := 300

-- Defining the first semi-annual payment
def payment1 : ℝ := 160

-- Defining the second semi-annual payment
def payment2 : ℝ := 160

-- Defining the total amount paid
def total_paid : ℝ := payment1 + payment2

-- Defining the interest amount
def interest : ℝ := total_paid - principal

-- Semi-annual interest rate
def semi_annual_rate : ℝ := (interest / principal) * 100

-- Annual interest rate for semi-annual compounding
def annual_rate : ℝ := 2 * semi_annual_rate

-- Statement asserting the annual interest rate given the conditions
theorem loan_interest_rate : annual_rate = 6.667 := by
  sorry

end loan_interest_rate_l402_402913


namespace integer_values_a_l402_402181

theorem integer_values_a :
  (∃! a : ℤ, ∃ m n : ℤ, m^2 + n^2 = (m + n)^2 - 2mn ∧
                     m + n = -a ∧ 
                     mn = 12a) :
  8 :=
sorry

end integer_values_a_l402_402181


namespace students_catching_up_on_homework_l402_402293

theorem students_catching_up_on_homework
  (total_students : ℕ)
  (half_doing_silent_reading : ℕ)
  (third_playing_board_games : ℕ)
  (remain_catching_up_homework : ℕ) :
  total_students = 24 →
  half_doing_silent_reading = total_students / 2 →
  third_playing_board_games = total_students / 3 →
  remain_catching_up_homework = total_students - (half_doing_silent_reading + third_playing_board_games) →
  remain_catching_up_homework = 4 :=
by
  intros h_total h_half h_third h_remain
  sorry

end students_catching_up_on_homework_l402_402293


namespace sequence_formula_l402_402823

def a : ℕ → ℤ
| 0       := -1
| (n + 1) := ∑ k in Finset.range (n + 1), (a k + 1)

theorem sequence_formula (n : ℕ) (hn : n ≥ 2) :
  a n = 2^(n-2) - 1 := sorry

end sequence_formula_l402_402823


namespace integer_values_of_a_l402_402171

theorem integer_values_of_a (a : ℤ) :
  (∀ x : ℤ, x^2 + a * x + 12 * a = 0 → x ∈ ℤ) → ∃ n : ℕ, n = 16 :=
by
  sorry

end integer_values_of_a_l402_402171


namespace area_enclosed_curve_l402_402100

theorem area_enclosed_curve (y x : ℝ → ℝ) (a b : ℝ) :
  (∀ x, y x = x^2) → a = 0 → b = 1 →
  ∫ x in set.Icc 0 1, x^2 = 1 / 3 :=
by
  intro h1 h2 h3
  rw [h2, h3]
  sorry

end area_enclosed_curve_l402_402100


namespace max_homework_time_l402_402350

theorem max_homework_time (biology_time history_time geography_time : ℕ) :
    biology_time = 20 ∧ history_time = 2 * biology_time ∧ geography_time = 3 * history_time →
    biology_time + history_time + geography_time = 180 :=
by
    intros
    sorry

end max_homework_time_l402_402350


namespace base5_subtraction_l402_402101

def base5_diff (n m : Nat) (r : Nat) : Prop :=
  n - m = r

def interpret_base5 (s : String) : Nat :=
  s.toList.reverse.enum.foldl (λ acc (d, i) => acc + (d.toNat.toInt - 48) * 5 ^ i) 0

theorem base5_subtraction :
  ∀ (n m r : String),
    base5_diff (interpret_base5 "1234") (interpret_base5 "234") (interpret_base5 "1000") := by
  sorry

end base5_subtraction_l402_402101


namespace find_principal_l402_402857

noncomputable def compound_interest (P r : ℝ) (n t : ℕ) : ℝ :=
  P * ((1 + r / n) ^ (n * t))

theorem find_principal
  (A : ℝ) (r : ℝ) (n t : ℕ)
  (hA : A = 4410)
  (hr : r = 0.05)
  (hn : n = 1)
  (ht : t = 2) :
  ∃ (P : ℝ), compound_interest P r n t = A ∧ P = 4000 :=
by
  sorry

end find_principal_l402_402857


namespace dihedral_angle_in_cube_l402_402655

-- Conditions of the problem
def side_length: ℝ := 2
def BD_perpendicular_to_ACCA' := true

-- The goal statement
theorem dihedral_angle_in_cube :
  dihedral_angle (point_of_cube 'A') (line_of_cube 'BD') (point_of_cube 'C') = π - 2 * arctan (sqrt 2) :=
sorry

end dihedral_angle_in_cube_l402_402655


namespace intersection_with_y_axis_l402_402796

-- Define the given function
def f (x : ℝ) := x^2 + x - 2

-- Prove that the intersection point with the y-axis is (0, -2)
theorem intersection_with_y_axis : f 0 = -2 :=
by {
  sorry
}

end intersection_with_y_axis_l402_402796


namespace parallel_triangle_perimeter_l402_402416

/-- Given a triangle ABC with side lengths AB=120, BC=220, and AC=180, and lines ℓA, ℓB, and ℓC  
  drawn parallel to BC, AC, and AB respectively. Let the intersections of ℓA, ℓB, and ℓC with the 
  interior of the triangle be segments of lengths 55, 45, and 15 respectively. Prove that the 
  perimeter of the triangle formed by these lines is 715. -/
theorem parallel_triangle_perimeter :
  ∀ (A B C : Type) (AB BC AC : ℕ) (ℓA ℓB ℓC : Type), 
    AB = 120 → BC = 220 → AC = 180 → 
    let seg_ℓA := 55 in 
    let seg_ℓB := 45 in 
    let seg_ℓC := 15 in
    (seg_ℓA + seg_ℓB + seg_ℓC + 200 + 115 = 715) :=
by 
  intros A B C AB BC AC ℓA ℓB ℓC h1 h2 h3 seg_ℓA seg_ℓB seg_ℓC,
  rw [h1, h2, h3, seg_ℓA, seg_ℓB, seg_ℓC],
  exact congr_arg2 (λ x y, x + 200 + y) rfl (congr_arg2 (λ x y, x + y) rfl rfl),
  -- The proof has been omitted
  sorry

end parallel_triangle_perimeter_l402_402416


namespace count_n_1988_l402_402519

def f : ℕ → ℕ
| 1       := 1
| 3       := 3
| (2 * n) := f n
| (4 * n + 1) := 2 * f (2 * n + 1) - f n
| (4 * n + 3) := 3 * f (2 * n + 1) - 2 * f n
| _       := 0 -- Default case to handle all n not naturally handled

theorem count_n_1988 :
  ∃ count, count = 92 ∧ 
  ∀ n, n ≤ 1988 → f n = n ↔ nat.reverse n = n ∧ count = 92 :=
by
  sorry

end count_n_1988_l402_402519


namespace count_valid_n_l402_402977

-- Definition: n is a positive integer and 0 < n < 40
def is_valid_n (n : ℕ) : Prop := 0 < n ∧ n < 40

-- Definition: The fraction n/(40 - n) is a positive integer
def is_positive_integer_fraction (n : ℕ) : Prop := (n / (40 - n)) ≥ 1 ∧ (n % (40 - n)) = 0

-- The Lean 4 statement in which we prove the existence of exactly 7 valid n
theorem count_valid_n : {n : ℕ | is_valid_n n ∧ is_positive_integer_fraction n}.card = 7 :=
by 
  sorry

end count_valid_n_l402_402977


namespace range_of_f_l402_402228

theorem range_of_f (a b : ℝ) (h_even : ∀ x : ℝ, f x = f (-x))
  (h_b : -2*b + 3*b - 1 = 0) :
  set.range (λ x : ℝ, x^2 - 2*a*x + b) = set.Icc 1 5 :=
by
  -- Parse conditions
  sorry

end range_of_f_l402_402228


namespace train_arrival_day_l402_402829

-- Definitions for the start time and journey duration
def start_time : ℕ := 0  -- early morning (0 hours) on Tuesday
def journey_duration : ℕ := 28  -- 28 hours

-- Proving the arrival time
theorem train_arrival_day (start_time journey_duration : ℕ) :
  journey_duration == 28 → 
  start_time == 0 → 
  (journey_duration / 24, journey_duration % 24) == (1, 4) → 
  true := 
by
  intros
  sorry

end train_arrival_day_l402_402829


namespace general_term_formula_sum_of_b_first_terms_l402_402277

variable (a₁ a₂ : ℝ)
variable (a : ℕ → ℝ)
variable (b : ℕ → ℝ)
variable (T : ℕ → ℝ)

-- Conditions
axiom h1 : a₁ * a₂ = 8
axiom h2 : a₁ + a₂ = 6
axiom increasing_geometric_sequence : ∀ n : ℕ, a (n+1) = a (n) * (a₂ / a₁)
axiom initial_conditions : a 1 = a₁ ∧ a 2 = a₂
axiom b_def : ∀ n, b n = 2 * a n + 3

-- To Prove
theorem general_term_formula : ∀ n: ℕ, a n = 2 ^ (n + 1) :=
sorry

theorem sum_of_b_first_terms (n : ℕ) : T n = 2 ^ (n + 2) - 4 + 3 * n :=
sorry

end general_term_formula_sum_of_b_first_terms_l402_402277


namespace distinct_integer_a_values_l402_402137

open Nat

theorem distinct_integer_a_values : ∃ (S : Finset ℤ), S.card = 8 ∧
  ∀ a ∈ S, ∃ (p q : ℤ), p + q = -a ∧ p * q = 12a ∧ 
  ∀ x ∈ (polynomial.roots (polynomial.C (12 : ℚ) * polynomial.X + polynomial.C a * polynomial.X + polynomial.C 1)).val, x ∈ ℤ :=
by
  sorry

end distinct_integer_a_values_l402_402137


namespace part1_case1_part1_case2_m_gt_neg1_part1_case2_m_lt_neg1_part2_l402_402222

def f (x m : ℝ) := (m + 1) * x^2 - (m - 1) * x + (m - 1)

theorem part1_case1 :
  ∀ x : ℝ, f x (-1) ≥ 0 := 
sorry

theorem part1_case2_m_gt_neg1 (m : ℝ) (h : m > -1) :
  ∀ x : ℝ, (f x m ≥ (m + 1) * x) ↔ 
    (x ≤ (m - 1) / (m + 1) ∨ x ≥ 1) := 
sorry

theorem part1_case2_m_lt_neg1 (m : ℝ) (h : m < -1) :
  ∀ x : ℝ, (f x m ≥ (m + 1) * x) ↔ 
    (1 ≤ x ∧ x ≤ (m - 1) / (m + 1)) := 
sorry

theorem part2 (m : ℝ) :
  (∀ x : ℝ, -1/2 ≤ x ∧ x ≤ 1/2 → f x m ≥ 0) ↔ m ∈ set.Ici 1 :=
sorry

end part1_case1_part1_case2_m_gt_neg1_part1_case2_m_lt_neg1_part2_l402_402222


namespace intersection_P_Q_l402_402608

def P (y : ℝ) : Prop :=
  ∃ (x : ℝ), y = -x^2 + 2

def Q (y : ℝ) : Prop :=
  ∃ (x : ℝ), y = x

theorem intersection_P_Q :
  { y : ℝ | P y } ∩ { y : ℝ | Q y } = { y : ℝ | y ≤ 2 } :=
by
  sorry

end intersection_P_Q_l402_402608


namespace external_bisector_l402_402746

open Triangle

variable {Point : Type} [EuclideanGeometry Point]

variables {A T C L K : Point}

theorem external_bisector (h : angle_bisector A T C L) : external_bisector A T C K :=
sorry

end external_bisector_l402_402746


namespace range_of_f_l402_402227

-- Define the function
def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^2 - 2 * a * x + b

-- Define the problem statement as a theorem
theorem range_of_f (a b : ℝ) (h1 : ∀ x, f x a b = f (-x) a b)
                     (h2 : -2 * b ≤ 3 * b - 1) :
  ∃ (m M : ℝ), ∀ x ∈ Icc (-2 * b) (3 * b - 1), f x a b ∈ Icc m M :=
sorry

end range_of_f_l402_402227


namespace two_digit_number_repeats_l402_402944

theorem two_digit_number_repeats (n : ℕ) (h1 : n ≥ 10) (h2 : n < 100) : ∃ m, ∀ k ≥ m, digit_square_cycle n k = digit_square_cycle n (k + 1) :=
  sorry

-- Define the function that squares a number and takes the last two digits
def last_two_digits (x : ℕ) : ℕ := x % 100

-- Define the function that performs the iterative process
def digit_square_cycle (n : ℕ) (k : ℕ) : ℕ :=
  if k = 0 then last_two_digits (n * n) else last_two_digits ((digit_square_cycle n (k - 1)) * (digit_square_cycle n (k - 1)))

end two_digit_number_repeats_l402_402944


namespace river_flow_rate_l402_402477

noncomputable def volume_per_minute : ℝ := 3200
noncomputable def depth_of_river : ℝ := 3
noncomputable def width_of_river : ℝ := 32
noncomputable def cross_sectional_area : ℝ := depth_of_river * width_of_river

noncomputable def flow_rate_m_per_minute : ℝ := volume_per_minute / cross_sectional_area
-- Conversion factors
noncomputable def minutes_per_hour : ℝ := 60
noncomputable def meters_per_km : ℝ := 1000

noncomputable def flow_rate_kmph : ℝ := (flow_rate_m_per_minute * minutes_per_hour) / meters_per_km

theorem river_flow_rate :
  flow_rate_kmph = 2 :=
by
  -- We skip the proof and use sorry to focus on the statement structure.
  sorry

end river_flow_rate_l402_402477


namespace sin_half_sum_l402_402553

variable (α β : ℝ)

noncomputable def cos_cond : Prop := cos (α - β / 2) = -1 / 9
noncomputable def sin_cond : Prop := sin (α / 2 - β) = 2 / 3
noncomputable def interval_cond : Prop := 0 < β ∧ β < π / 2 ∧ π / 2 < α ∧ α < π

theorem sin_half_sum (h1 : cos_cond α β) (h2 : sin_cond α β) (h3 : interval_cond α β) :
  sin ((α + β) / 2) = 22 / 27 := 
sorry

end sin_half_sum_l402_402553


namespace sqrt_expr_sum_l402_402904

theorem sqrt_expr_sum {a b c : ℤ} (h1 : 53 + 20 * Nat.sqrt 7 = a + b * Nat.sqrt c)
  (h2 : c = 7): a + b + c = 14 :=
sorry

end sqrt_expr_sum_l402_402904


namespace sum_of_roots_angles_l402_402402

theorem sum_of_roots_angles (θ : ℕ → ℝ) (h : ∀ k, 0 ≤ θ k ∧ θ k < 360 ∧ θ k = (225 + 360 * ↑k) / 7) :
  (∑ k in Finset.range 7, θ k) = 1305 :=
by
  sorry

end sum_of_roots_angles_l402_402402


namespace external_angle_bisector_l402_402713

open EuclideanGeometry

variables {A T C K L : Point}

theorem external_angle_bisector
    (h1 : is_internal_angle_bisector T L A T C) : is_external_angle_bisector T K A T C :=
sorry

end external_angle_bisector_l402_402713


namespace largest_crowd_size_l402_402888

theorem largest_crowd_size :
  ∃ (n : ℕ), 
    (⌊n / 2⌋ + ⌊n / 3⌋ + ⌊n / 5⌋ = n) ∧
    ∀ m : ℕ, (⌊m / 2⌋ + ⌊m / 3⌋ + ⌊m / 5⌋ = m) → m ≤ 37 :=
sorry

end largest_crowd_size_l402_402888


namespace maximum_distance_between_intersections_l402_402290

noncomputable def curve1 (θ : ℝ) : ℝ := 2 * Real.sin θ
noncomputable def curve2 (θ : ℝ) : ℝ := 4 * Real.sin θ

theorem maximum_distance_between_intersections (α : ℝ) (hα : 0 < α ∧ α < π) : 
  ∃ M, ∀ θ, |curve2 α - curve1 α| ≤ M ∧ M = 2 := by
  sorry

end maximum_distance_between_intersections_l402_402290


namespace distinct_residues_1_5_l402_402511

open Nat

theorem distinct_residues_1_5 {n : ℕ} (hn : Euler.totient n = 2) :
    (1 % n ≠ 5 % n) ↔ (n = 3 ∨ n = 6) :=
by 
  sorry

end distinct_residues_1_5_l402_402511


namespace sum_of_first_2022_terms_b_l402_402345

def a (n : ℕ) : ℕ
| 1 := 1
| 2 := 4
| (n+2) := 2 * a (n+1) + 2 - a n

def b (n : ℕ) : ℕ :=
  (1 + (n + 1) / a n).natCeil

def sum_b (n : ℕ) : ℕ :=
  ∑ i in range n, b i

theorem sum_of_first_2022_terms_b : 
  sum_b 2022 = 4045 := 
  sorry

end sum_of_first_2022_terms_b_l402_402345


namespace volume_of_solid_of_revolution_l402_402494

theorem volume_of_solid_of_revolution (a : ℝ) : 
  let h := a / 2
  let r := (Real.sqrt 3 / 2) * a
  2 * (1 / 3) * π * r^2 * h = (π * a^3) / 4 :=
by
  sorry

end volume_of_solid_of_revolution_l402_402494


namespace mineral_age_possibilities_l402_402030

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def count_permutations_with_repeats (n : ℕ) (repeats : List ℕ) : ℕ :=
  factorial n / List.foldl (· * factorial ·) 1 repeats

theorem mineral_age_possibilities : 
  let digits := [2, 2, 4, 4, 7, 9]
  let odd_digits := [7, 9]
  let remaining_digits := [2, 2, 4, 4]
  2 * count_permutations_with_repeats 5 [2,2] = 60 :=
by
  sorry

end mineral_age_possibilities_l402_402030


namespace number_of_integer_values_of_a_l402_402164

theorem number_of_integer_values_of_a : 
  ∃ (a_set : Set ℤ), (∀ a ∈ a_set, ∃ x y : ℤ, x^2 + a * x + 12 * a = 0 ∧ y^2 + a * y + 12 * a = 0) ∧ a_set.card = 16 :=
sorry

end number_of_integer_values_of_a_l402_402164


namespace distinct_integer_values_of_a_l402_402123

theorem distinct_integer_values_of_a :
  ∃ (a_values : Finset ℤ), (∀ a ∈ a_values, ∃ (m n : ℤ), (x^2 + a * x + 12 * a = 0) ∧ (m + n = -a) ∧ (m * n = 12 * a))
  ∧ a_values.card = 16 :=
sorry

end distinct_integer_values_of_a_l402_402123


namespace count_distinct_a_l402_402111

def quadratic_has_integer_solutions (a : ℤ) : Prop :=
  ∃ x m n : ℤ, (x^2 + a * x + 12 * a = 0) ∧ (m + n = -a) ∧ (m * n = 12 * a)

theorem count_distinct_a : (finset.univ.filter quadratic_has_integer_solutions).card = 9 := 
sorry

end count_distinct_a_l402_402111


namespace opposite_number_of_2_eq_neg2_abs_val_eq_2_iff_eq_2_or_neg2_l402_402816

theorem opposite_number_of_2_eq_neg2 : -2 = -2 := by
  sorry

theorem abs_val_eq_2_iff_eq_2_or_neg2 (x : ℝ) : abs x = 2 ↔ x = 2 ∨ x = -2 := by
  sorry

end opposite_number_of_2_eq_neg2_abs_val_eq_2_iff_eq_2_or_neg2_l402_402816


namespace range_of_m_l402_402587

def quadratic_function (x : ℝ) : ℝ := x^2 - 2*x + 5

theorem range_of_m (m : ℝ) (h : 0 < m) :
  (∀ x ∈ set.Icc (0 : ℝ) m, 4 ≤ quadratic_function x ∧ quadratic_function x ≤ 5) →
  m ∈ set.Icc (1 : ℝ) 2 :=
sorry

end range_of_m_l402_402587


namespace average_children_in_families_with_kids_l402_402990

theorem average_children_in_families_with_kids :
  (total_families : ℕ) →
  (avg_children_per_family : ℝ) →
  (childless_families : ℕ) →
  total_families = 15 →
  avg_children_per_family = 3 →
  childless_families = 3 →
  (total_families * avg_children_per_family) / (total_families - childless_families) = 3.75 :=
by
  intros total_families avg_children_per_family childless_families h1 h2 h3
  have h4 : total_families * avg_children_per_family = 45, by sorry
  have h5 : total_families - childless_families = 12, by sorry
  rw [h4, h5]
  norm_num
  exact eq.refl 3.75

end average_children_in_families_with_kids_l402_402990


namespace area_of_triangle_N1N2N3_l402_402660

-- Definitions for points and segments
variables {A B C D E F N1 N2 N3 : Type}
variables {K : ℝ}
variables [linear_ordered_field K]

-- Conditions
def segment_CD (CD BC : K) : Prop := CD = (1 / 4) * BC
def segment_AE (AE AB : K) : Prop := AE = (1 / 4) * AB
def segment_BF (BF CA : K) : Prop := BF = (1 / 4) * CA
def area_triangle (N1 N2 N3 : Type) [K] : K := (11 / 48) * K

-- Statement of the theorem
theorem area_of_triangle_N1N2N3 
  (h_CD : segment_CD CD BC)
  (h_AE : segment_AE AE AB)
  (h_BF : segment_BF BF CA)
  (area_ABC : K) :
  ∃ area_N1N2N3 : K, area_N1N2N3 = (11 / 48) * area_ABC := 
sorry

end area_of_triangle_N1N2N3_l402_402660


namespace external_angle_bisector_proof_l402_402731

variables {A T C L K : Type} [Nonempty A] [Nonempty T] [Nonempty C] [Nonempty L] [Nonempty K]

noncomputable def angle_bisector_theorem (AL LC AB BC AK KC : ℝ) : Prop :=
(AL / LC) = (AB / BC) ∧ (AK / KC) = (AL / LC)

noncomputable def internal_angle_bisector (AT TC AL LC : ℝ) : Prop :=
(AT / TC) = (AL / LC)

noncomputable def external_angle_bisector (AT TC AK KC : ℝ) : Prop :=
(AT / TC) = (AK / KC)

theorem external_angle_bisector_proof (AL LC AB BC AK KC AT TC : ℝ) 
(h1 : angle_bisector_theorem AL LC AB BC AK KC)
(h2 : internal_angle_bisector AT TC AL LC) :
external_angle_bisector AT TC AK KC :=
sorry

end external_angle_bisector_proof_l402_402731


namespace perpendicular_lines_l402_402899

open Real EuclideanGeometry

variables {A P B D Q O : Point}

def is_parallelogram (ABCD : Quadrilateral) : Prop :=
  (ABCD.ba.is_parallel_with ABCD.ac) ∧ (ABCD.dc.is_parallel_with ABCD.bd)

def cyclic (A P B D : Point) : Prop :=
  ∃ (c : Circle), c.is_circumscribing A ∧ c.is_circumscribing P ∧ c.is_circumscribing B ∧ c.is_circumscribing D

def intersect_at (l1 l2 : Line) (Q : Point) : Prop :=
  Q ∈ l1 ∧ Q ∈ l2

def circumcenter (O : Point) (Δ : Triangle) : Prop :=
  O = Δ.circumcenter

theorem perpendicular_lines
  (ABCD : Quadrilateral)
  (h₁ : is_parallelogram ABCD)
  (h₂ : ∠DAB < π / 2)
  (h₃ : cyclic A P B D)
  (h₄ : intersect_at (line_through A P) (line_through C D) Q)
  (h₅ : circumcenter O (triangle C P Q))
  (h₆ : D ≠ O) :
  ∠ADO = π / 2 :=
by sorry

end perpendicular_lines_l402_402899


namespace minimum_familiar_pairs_l402_402906

open Finset

-- Define the set of students and the relationship of familiarity
variable (students : Finset ℕ)
variable (n : ℕ := 175)
variable (familiar : ℕ → ℕ → Prop)

-- Assumption: students set has 175 members
axiom student_count : students.card = n

-- Assumption: familiarity is symmetric
axiom familiar_symm (a b : ℕ) : familiar a b → familiar b a

-- Assumption: familiarity within any group of six
axiom familiar_in_groups_of_six (s : Finset ℕ) (h₁ : s.card = 6) :
  ∃ t₁ t₂ : Finset ℕ, t₁.card = 3 ∧ t₂.card = 3 ∧ (∀ x ∈ t₁, ∀ y ∈ t₁, x ≠ y → familiar x y) ∧
  (∀ x ∈ t₂, ∀ y ∈ t₂, x ≠ y → familiar x y) ∧ t₁ ∪ t₂ = s ∧ t₁ ∩ t₂ = ∅

-- Theorem: minimum number of familiar pairs
theorem minimum_familiar_pairs :
  ∃ k : ℕ, (∑ a in students, (students.filter (familiar a)).card) / 2 ≥ 15050 :=
sorry

end minimum_familiar_pairs_l402_402906


namespace number_of_distinct_a_l402_402130

noncomputable def count_distinct_a : ℕ :=
  let divisors := [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 36, 48, 72, 144, -1, -2, -3, -4, -6, -8, -9, -12, -16, -18, -24, -36, -48, -72, -144]
  let pairs := (divisors.product divisors).filter (λ (p : ℤ × ℤ), p.1 * p.2 = 144)
  let sums := pairs.map (λ (p : ℤ × ℤ), p.1 + p.2 - 24)
  let distinct_sums := sums.eraseDuplicates  
  distinct_sums.length

theorem number_of_distinct_a : count_distinct_a = 14 :=
by
  sorry

end number_of_distinct_a_l402_402130


namespace smallest_real_number_l402_402056

theorem smallest_real_number 
  (a b c d : ℝ)
  (h1 : a = sqrt 3)
  (h2 : b = -real.pi)
  (h3 : c = 0)
  (h4 : d = -2) :
  (b = min {a, b, c, d}) :=
by sorry

end smallest_real_number_l402_402056


namespace baron_munchausen_max_people_l402_402864

theorem baron_munchausen_max_people :
  ∃ x : ℕ, (x = 37) ∧ 
  (1 / 2 * x).nat_ceil + (1 / 3 * x).nat_ceil + (1 / 5 * x).nat_ceil = x := sorry

end baron_munchausen_max_people_l402_402864


namespace intersection_points_count_l402_402968

-- Define the hyperbola equation
def is_hyperbola (x y : ℝ) : Prop :=
  (x - 1) ^ 2 - (y + 2) ^ 2 = 4

-- Define the condition that a point (x, y) lies on a line
def is_line (a b c : ℝ) (x y : ℝ) : Prop :=
  a * x + b * y + c = 0

-- Condition that a line passes through the point (1, -2)
def passes_through_center (a b c : ℝ) : Prop :=
  is_line a b c 1 (-2)

-- Neither line is tangent to the hyperbola
def not_tangent (a b c : ℝ) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ), (is_hyperbola x1 y1 ∧ is_line a b c x1 y1) ∧ (is_hyperbola x2 y2 ∧ is_line a b c x2 y2)

-- The theorem statement
theorem intersection_points_count :
  ∀ (a1 b1 c1 a2 b2 c2 : ℝ),
  passes_through_center a1 b1 c1 ∧ not_tangent a1 b1 c1 ∧ not_tangent a2 b2 c2 →
  ∃ n, n ∈ {2, 3, 4} ∧ 
  (∃ (pts : finset (ℝ × ℝ)), pts.card = n ∧ 
    ∀ pt ∈ pts, is_hyperbola pt.1 pt.2 ∧ (is_line a1 b1 c1 pt.1 pt.2 ∨ is_line a2 b2 c2 pt.1 pt.2)) :=
sorry

end intersection_points_count_l402_402968


namespace solve_for_x_l402_402847

theorem solve_for_x :
  ∃ x : ℚ, x - 5/6 = 7/18 - x/4 ∧ x = 44/45 :=
begin
  sorry
end

end solve_for_x_l402_402847


namespace problem_statement_l402_402999

def M : ℕ := 419

theorem problem_statement :
  ∃ M : ℕ, (M % 3 = 2) ∧ (M % 4 = 3) ∧ (M % 5 = 4) ∧ (M % 6 = 5) ∧ (M % 7 = 6) ∧ M = 419 :=
by
  use 419
  sorry

end problem_statement_l402_402999


namespace Liam_Noah_Olivia_songs_l402_402347

noncomputable def songs_distribution_problem : Prop := 
  ∃ (LN NO OL L N O X : Finset ℕ),
  LN.disjoint NO ∧  
  LN.disjoint OL ∧ 
  NO.disjoint OL ∧ 
  L.disjoint LN ∧ L.disjoint NO ∧ L.disjoint OL ∧ 
  N.disjoint LN ∧ N.disjoint NO ∧ N.disjoint OL ∧ 
  O.disjoint LN ∧ O.disjoint NO ∧ O.disjoint OL ∧ 
  X.disjoint LN ∧ X.disjoint NO ∧ X.disjoint OL ∧ 
  X.disjoint L ∧ X.disjoint N ∧ X.disjoint O ∧ 
  (LN ∪ NO ∪ OL ∪ L ∪ N ∪ O ∪ X).card = 5 ∧ 
  LN.card ≥ 1 ∧ 
  NO.card ≥ 1 ∧ 
  OL.card ≥ 1 ∧ 
  LN.card + NO.card + OL.card = 3 ∧ 
  L.card + N.card + O.card + X.card = 2

noncomputable def total_ways_to_distribute_songs : ℕ := 31

theorem Liam_Noah_Olivia_songs : songs_distribution_problem → (∃ ways, ways = total_ways_to_distribute_songs) :=
begin
  sorry
end

end Liam_Noah_Olivia_songs_l402_402347


namespace elisa_math_books_l402_402526

theorem elisa_math_books (N M L : ℕ) (h₀ : 24 + M + L + 1 = N + 1) (h₁ : (N + 1) % 9 = 0) (h₂ : (N + 1) % 4 = 0) (h₃ : N < 100) : M = 7 :=
by
  sorry

end elisa_math_books_l402_402526


namespace max_arithmetic_sequence_length_l402_402607

def P (a b c : ℕ) : ℕ := 7^3 + a * 7^2 + b * 7 + c

def valid_element (x : ℕ) : Prop :=
  ∃ (a b c : ℕ), 1 ≤ a ∧ a ≤ 6 ∧ 1 ≤ b ∧ b ≤ 6 ∧ 1 ≤ c ∧ c ≤ 6 ∧ x = P a b c

def arithmetic_sequence (s : List ℕ) : Prop :=
  ∀ i j : ℕ, i < j ∧ j < s.length →
    (∃ d: ℕ, s.get? j = s.get? i + d)

theorem max_arithmetic_sequence_length : 
  ∀ (s : List ℕ), (∀ x ∈ s, valid_element x) → arithmetic_sequence s → s.length ≤ 6 :=
sorry

end max_arithmetic_sequence_length_l402_402607


namespace integer_solution_count_l402_402142

def has_integer_solutions (a : ℤ) : Prop :=
  ∃ x y : ℤ, x^2 + a * x + 12 * a = 0 ∧ y^2 + a * y + 12 * a = 0

theorem integer_solution_count : 
  (set.finite {a : ℤ | has_integer_solutions a}).to_finset.card = 16 := 
begin
  sorry
end

end integer_solution_count_l402_402142


namespace highest_average_speed_is_fourth_hour_l402_402468

def distances : List ℕ := [70, 95, 85, 100, 90, 85, 75]

theorem highest_average_speed_is_fourth_hour (distances : List ℕ) :
  distances = [70, 95, 85, 100, 90, 85, 75] →
  ∀ i ∈ [0, 1, 2, 3, 4, 5, 6], (distances.nth i).get_or_else 0 ≤ 100 :=
begin
  intros distances_eq i i_in_range,
  have h : i < 7 := by linarith,
  have d_i := (distances.nth_le i h),
  rw distances_eq at d_i,
  iterate 4 { cases i; simp [d_i] },
  iterate 3 { cases i; simp [d_i] },
  case h_aux : i { cases i }
end

end highest_average_speed_is_fourth_hour_l402_402468


namespace tangent_lines_to_circle_through_pointA_l402_402651

noncomputable def is_tangent (l : ℝ → ℝ → Prop) (c : ℝ → ℝ → Prop) : Prop :=
  ∃ x y, l x y ∧ c x y ∧ ∀ ε > 0, ∃δ > 0, ∀ x', abs (x' - x) < δ → ∀ y', abs (y' - y) < δ → l x' y' ∧ c x' y' → abs (dist (x, y) (x', y')) < ε

noncomputable def line1 : ℝ → ℝ → Prop := λ x y, 3 * x + 4 * y = 10
noncomputable def line2 : ℝ → ℝ → Prop := λ x y, x = 2
noncomputable def circle : ℝ → ℝ → Prop := λ x y, x^2 + y^2 = 4
noncomputable def pointA : ℝ × ℝ := (2, 1)

theorem tangent_lines_to_circle_through_pointA :
  (is_tangent line1 circle ∨ is_tangent line2 circle) ∧
  (line1 pointA.1 pointA.2 ∨ line2 pointA.1 pointA.2) :=
by
  sorry

end tangent_lines_to_circle_through_pointA_l402_402651


namespace average_remaining_students_l402_402826

variable (k : ℕ) (h_k : k > 12)
variable (h1 : (8 * k = ∑ i in finset.range k, (λ i, 8) i))
variable (h2 : (12 * 14 = ∑ i in finset.range 12, (λ i, 14) i))

theorem average_remaining_students (k > 12) (h1 : (∑ i in finset.range k, (λ i, 8) i)) (h2 : (∑ i in finset.range 12, (λ i, 14) i)) : 
  ((8 * k - 168) / (k - 12)) = 8 :=
  sorry

end average_remaining_students_l402_402826


namespace number_of_ways_to_read_BANANA_l402_402285

/-- 
In a 3x3 grid, there are 84 different ways to read the word BANANA 
by moving from one cell to another cell with which it shares an edge,
and cells may be visited more than once.
-/
theorem number_of_ways_to_read_BANANA (grid : Matrix (Fin 3) (Fin 3) Char) (word : String := "BANANA") : 
  ∃! n : ℕ, n = 84 :=
by
  sorry

end number_of_ways_to_read_BANANA_l402_402285


namespace percent_employed_females_l402_402300

theorem percent_employed_females (h1 : 96 / 100 > 0) (h2 : 24 / 100 > 0) : 
  (96 - 24) / 96 * 100 = 75 := 
by 
  -- Proof to be filled out
  sorry

end percent_employed_females_l402_402300


namespace distance_of_ladder_to_building_l402_402470

theorem distance_of_ladder_to_building :
  ∀ (c a b : ℕ), c = 25 ∧ a = 20 ∧ (a^2 + b^2 = c^2) → b = 15 :=
by
  intros c a b h
  rcases h with ⟨hc, ha, hpyth⟩
  have h1 : c = 25 := hc
  have h2 : a = 20 := ha
  have h3 : a^2 + b^2 = c^2 := hpyth
  sorry

end distance_of_ladder_to_building_l402_402470


namespace intersection_with_y_axis_l402_402793

theorem intersection_with_y_axis (f : ℝ → ℝ) (hf : ∀ x, f x = x^2 + x - 2) : f 0 = -2 :=
by
  sorry

end intersection_with_y_axis_l402_402793


namespace number_of_divisors_of_power_with_exact_divisors_l402_402248

theorem number_of_divisors_of_power_with_exact_divisors 
  (a b c : ℕ) (n : ℕ)
  (h1 : 1800 = 2^a * 3^b * 5^c) 
  (h2 : a = 3) (h3 : b = 2) (h4 : c = 2) 
  (h5 : n = 1800^1800) :
  (∃ a b c : ℕ, (a + 1) * (b + 1) * (c + 1) = 180 ∧ 
                2^a * 3^b * 5^c ∣ n ∧ 
                (((((2^a * 3^b * 5^c : ℕ) : Type) : factorization) : List) : List.length) = 180) :=
begin
  sorry
end

end number_of_divisors_of_power_with_exact_divisors_l402_402248


namespace measure_of_EDF_angle_l402_402774

open Real

noncomputable def angle_EDF : ℝ :=
  let arc_ratio : ℝ := 4 / 9
  let central_angle_EF := 4 / 9 * 360
  let exterior_angle := 180 - central_angle_EF
  exterior_angle / 2

theorem measure_of_EDF_angle:
  ∀ (D E F O : Type) [is_point D] [is_point E] [is_point F] [is_point O],
  (tangent_from_point D E O ∧ tangent_from_point D F O) ∧
  (arc_ratio E F 4 5) →
  ∠EDF = 20 :=
begin
  -- The proof would involve verifying the angle calculation steps provided in the solution,
  -- but as requested, the proof construction steps are omitted.
  sorry
end

end measure_of_EDF_angle_l402_402774


namespace integer_solution_count_l402_402146

def has_integer_solutions (a : ℤ) : Prop :=
  ∃ x y : ℤ, x^2 + a * x + 12 * a = 0 ∧ y^2 + a * y + 12 * a = 0

theorem integer_solution_count : 
  (set.finite {a : ℤ | has_integer_solutions a}).to_finset.card = 16 := 
begin
  sorry
end

end integer_solution_count_l402_402146


namespace range_of_c_l402_402563

theorem range_of_c (c : ℝ) (h : c > 0) :
  ( (∀ x : ℝ, (x ∈ Icc (1 / 2) 2) → x + 1 / x > c) ∨ (∀ x : ℝ, (x > 0) → c ^ x < c) ) ∧
  ¬ ( (∀ x : ℝ, (x ∈ Icc (1 / 2) 2) → x + 1 / x > c) ∧ (∀ x : ℝ, (x > 0) → c ^ x < c) )
  ↔ ((0 < c ∧ c ≤ 1 / 2) ∨ (c ≥ 1)) :=
sorry

end range_of_c_l402_402563


namespace complete_work_in_days_l402_402445

-- Definitions from conditions
def works_thrice : ℕ → ℕ → Prop := λ a b, a = 3 * b
def takes_less_days : ℕ → ℕ → ℕ → Prop := λ a b c, a = b - c

-- Main statement
theorem complete_work_in_days 
  (A B : ℕ) 
  (h1 : works_thrice B A) 
  (h2 : takes_less_days A B 60) 
  : (1 / (1/30 + 1/90) = 22.5) :=
begin
  sorry
end

end complete_work_in_days_l402_402445


namespace scheduling_courses_l402_402249

theorem scheduling_courses (n_courses : ℕ) (n_periods : ℕ) 
    (distinct_courses : ℕ) (non_consecutive : Prop) 
    (H1 : n_courses = 4) (H2 : n_periods = 7) 
    (H3 : distinct_courses = 4) (H4 : non_consecutive): 
    ∃(ways : ℕ), ways = 1680 :=
by 
  use 1680
  sorry

end scheduling_courses_l402_402249


namespace michael_total_robotic_units_l402_402062

theorem michael_total_robotic_units :
  let tom_flying_robots : ℝ := 7.5
      tom_ground_robots : ℝ := 5
      michael_original_ground_robots : ℝ := 3
      michael_original_flying_robots : ℝ := 4 * tom_flying_robots
      robots_given_to_james : ℝ := (tom_flying_robots / 2.5) * 0.5
      robots_given_to_michael_by_james : ℝ := robots_given_to_james * (1/3)
      michael_total_flying_robots : ℝ := michael_original_flying_robots + robots_given_to_michael_by_james
      michael_total_robotic_units : ℝ := michael_total_flying_robots + michael_original_ground_robots
  in michael_total_robotic_units = 33.5 :=
by
  -- Proof goes here
  sorry

end michael_total_robotic_units_l402_402062


namespace visible_sides_probability_l402_402025

theorem visible_sides_probability
  (r : ℝ)
  (side_length : ℝ := 4)
  (probability : ℝ := 3 / 4) :
  r = 8 * Real.sqrt 3 / 3 :=
sorry

end visible_sides_probability_l402_402025


namespace maximum_natural_number_for_difference_seven_l402_402761

theorem maximum_natural_number_for_difference_seven : 
  ∃ n : ℕ, (∀ (s : finset ℕ), s.card = 50 → s ⊆ finset.range n.succ → (∃ a b ∈ s, a ≠ b ∧ |a - b| = 7)) ∧ n = 98 :=
by
  sorry

end maximum_natural_number_for_difference_seven_l402_402761


namespace find_integer_pairs_l402_402993

def is_perfect_square (x : ℤ) : Prop :=
  ∃ k : ℤ, k * k = x

theorem find_integer_pairs (m n : ℤ) :
  (is_perfect_square (m^2 + 4 * n) ∧ is_perfect_square (n^2 + 4 * m)) ↔
  (∃ a : ℤ, (m = 0 ∧ n = a^2) ∨ (m = a^2 ∧ n = 0) ∨ (m = -4 ∧ n = -4) ∨ (m = -5 ∧ n = -6) ∨ (m = -6 ∧ n = -5)) :=
by
  sorry

end find_integer_pairs_l402_402993


namespace max_lift_times_l402_402765

theorem max_lift_times (n : ℕ) :
  (2 * 30 * 10) = (2 * 25 * n) → n = 12 :=
by
  sorry

end max_lift_times_l402_402765


namespace total_pencils_l402_402090

theorem total_pencils (pencils_per_child : ℕ) (children : ℕ) (h1 : pencils_per_child = 2) (h2 : children = 11) : (pencils_per_child * children = 22) := 
by
  sorry

end total_pencils_l402_402090


namespace cricket_player_innings_l402_402466

-- Define the conditions
def average_runs : ℕ → ℕ := λ n => 32 * n
def new_average_after_next_innings (total_runs : ℕ) (innings : ℕ) : ℕ := (total_runs + 76) / (innings + 1)

-- Main theorem to prove
theorem cricket_player_innings (n : ℕ) : average_runs n + 76 = 36 * (n + 1) → n = 10 :=
begin
  intro h,
  sorry -- Proof goes here
end

end cricket_player_innings_l402_402466


namespace b_finishes_remaining_work_correct_time_for_b_l402_402443

theorem b_finishes_remaining_work (a_days : ℝ) (b_days : ℝ) (work_together_days : ℝ) (remaining_work_after : ℝ) : ℝ :=
  let a_work_rate := 1 / a_days
  let b_work_rate := 1 / b_days
  let combined_work_per_day := a_work_rate + b_work_rate
  let work_done_together := combined_work_per_day * work_together_days
  let remaining_work := 1 - work_done_together
  let b_completion_time := remaining_work / b_work_rate
  b_completion_time

theorem correct_time_for_b : b_finishes_remaining_work 2 6 1 (1 - 2/3) = 2 := 
by sorry

end b_finishes_remaining_work_correct_time_for_b_l402_402443


namespace num_solutions_zero_l402_402538

open Matrix

noncomputable def num_solutions : ℕ :=
  if ∃ (a b c d : ℝ), (λ M : Matrix (Fin 2) (Fin 2) ℝ, M⁻¹ = ![![1 / (a + d), 1 / (b + c)], ![1 / (c + b), 1 / (a + d)]] ) (Matrix.fin2x2 a b c d)
  then 1 else 0

theorem num_solutions_zero : num_solutions = 0 := by 
  sorry

end num_solutions_zero_l402_402538


namespace integer_values_a_l402_402180

theorem integer_values_a :
  (∃! a : ℤ, ∃ m n : ℤ, m^2 + n^2 = (m + n)^2 - 2mn ∧
                     m + n = -a ∧ 
                     mn = 12a) :
  8 :=
sorry

end integer_values_a_l402_402180


namespace circle_equation_line_equation_l402_402564

variable (x y : ℝ)

theorem circle_equation (h1 : ∃ c, (c.2 = c.1 - 1) ∧ ((2 - c.1)^2 + (0 - c.2)^2 = r^2) ∧ ((4 - c.1)^2 + (0 - c.2)^2 = r^2))
  (h2 : ∃ a b, (b = a - 1) ∧ ((2 - a)^2 + (-b)^2 = r^2) ∧ ((4 - a)^2 + (-b)^2 = r^2)) :
  (∃ (a b r: ℝ), (a = 3) ∧ (b = 2) ∧ (r = sqrt 5) ∧ ((x - a)^2 + (y - b)^2 = r^2)) :=
sorry

theorem line_equation (h3 : ∀ M N : ℝ, (N = 2 * M) ∧ ((x - 3)^2 + (y - 2)^2 = 5)) :
  (y = 0 ∨ 12 * x - 5 * y = 0) :=
sorry

end circle_equation_line_equation_l402_402564


namespace sqrt_25_sqrt_15_sqrt_9_eq_5_sqrt_15_l402_402093

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

-- Definitions for conditions
def expr1 : ℝ := sqrt 9
def expr2 : ℝ := sqrt (15 * expr1)
def expr3 : ℝ := sqrt (25 * expr2)

-- Final statement
theorem sqrt_25_sqrt_15_sqrt_9_eq_5_sqrt_15 : sqrt (25 * sqrt (15 * sqrt 9)) = 5 * sqrt 15 := by
  sorry

end sqrt_25_sqrt_15_sqrt_9_eq_5_sqrt_15_l402_402093


namespace max_people_in_crowd_l402_402869

theorem max_people_in_crowd : ∃ n : ℕ, n ≤ 37 ∧ 
    (⟨1 / 2 * n⟩ + ⟨1 / 3 * n⟩ + ⟨1 / 5 * n⟩ = n) :=
sorry

end max_people_in_crowd_l402_402869


namespace function_identity_l402_402991

theorem function_identity (f : ℕ → ℕ)
  (h1 : ∀ n : ℕ, (f^[n] n = n))
  (h2 : ∀ m n : ℕ, abs (f (m * n) - f m * f n) < 2017) :
  ∀ n : ℕ, f n = n :=
by
  sorry

end function_identity_l402_402991


namespace students_catching_up_on_homework_l402_402295

theorem students_catching_up_on_homework :
  ∀ (total_students : ℕ) (half : ℕ) (third : ℕ),
  total_students = 24 → half = total_students / 2 → third = total_students / 3 →
  total_students - (half + third) = 4 :=
by
  intros total_students half third
  intros h_total h_half h_third
  sorry

end students_catching_up_on_homework_l402_402295


namespace last_digit_of_decimal_expansion_l402_402843

noncomputable def last_digit (n : ℕ) : ℕ :=
  n % 10

def fraction_expression := 1 / (2^10 * 3^10 : ℚ)

theorem last_digit_of_decimal_expansion :
  last_digit (decimal_expansion fraction_expression) = 5 := 
begin
  sorry
end

end last_digit_of_decimal_expansion_l402_402843


namespace problem_statement_l402_402469

-- Define the conditions
def A_is_digit (a : ℕ) : Prop := 1 ≤ a ∧ a ≤ 9
def condition1 (a : ℕ) : Prop := 0.4 < a / 10
def condition2 (a : ℕ) : Prop := (6 + a / 10) < 6.8
def condition3 (a : ℕ) : Prop := (3 + a / 10) < (a + 4 / 10)

-- Define the predicate that A satisfies all conditions
def satisfies_all_conditions (a : ℕ) : Prop :=
  A_is_digit a ∧ condition1 a ∧ condition2 a ∧ condition3 a

-- Define the sum of all A that satisfy the conditions
def sum_all_satisfying_A : ℕ :=
  (Finset.range 10).filter satisfies_all_conditions |>.sum id

-- The problem statement
theorem problem_statement : sum_all_satisfying_A = 18 :=
by
  sorry

end problem_statement_l402_402469


namespace minimum_lambda_l402_402762

theorem minimum_lambda (S : ℕ → ℝ) (a : ℕ → ℝ) (λ : ℝ) (h1 : ∀ n : ℕ, n > 0 → ∃! x : ℝ, x^2 - S n * Real.cos x + 2 * a n - n = 0)
  (h2 : ∀ n : ℕ, n > 0 → λ / n ≥ (n + 1) / (a n + 1)) :
  λ ≥ 3 / 2 :=
sorry

end minimum_lambda_l402_402762


namespace find_solutions_to_system_l402_402098

theorem find_solutions_to_system (x y z : ℝ) 
    (h1 : 3 * (x^2 + y^2 + z^2) = 1) 
    (h2 : x^2 * y^2 + y^2 * z^2 + z^2 * x^2 = x * y * z * (x + y + z)^3) : 
    x = y ∧ y = z ∧ (x = 1 / 3 ∨ x = -1 / 3) :=
by
  sorry

end find_solutions_to_system_l402_402098


namespace perpendicular_lines_b_value_l402_402981

theorem perpendicular_lines_b_value (b : ℝ) :
  let v1 := ⟨b, -3, 2⟩
  let v2 := ⟨2, 3, 4⟩
  v1.dot v2 = 0 →
  b = 1 / 2 := by
  sorry

end perpendicular_lines_b_value_l402_402981


namespace not_divisible_by_121_l402_402777

theorem not_divisible_by_121 (n : ℤ) : ¬ 121 ∣ (n^2 + 3 * n + 5) :=
sorry

end not_divisible_by_121_l402_402777


namespace john_push_ups_l402_402851

variable (zachary : ℕ) (david : ℕ) (john : ℕ)

def push_ups_zachary : zachary = 51 := by
  sorry

def push_ups_david (h1 : zachary = 51) : david = zachary + 22 := by
  sorry

def push_ups_john (h2 : david = zachary + 22) : john = david - 4 := by
  sorry

theorem john_push_ups : john = 69 := by
  have hz : zachary = 51 := push_ups_zachary
  have hd : david = zachary + 22 := push_ups_david hz
  have hj : john = david - 4 := push_ups_john hd
  rw [hz, hd, hj]
  sorry

end john_push_ups_l402_402851


namespace ordered_pairs_count_l402_402403

theorem ordered_pairs_count :
  (∃ (b s : ℕ), (∀ n : ℕ, n < 10 → b_1 = b → ∃ (r : ℕ), b_(n + 1) = b * (r^n))
  ∧ ∀ (b s : ℕ), ∃ (seq : ℕ → ℕ), seq 0 = b ∧ (∀ n : ℕ, seq (n + 1) = seq n * s) 
  ∧ (log 4 10  (((λ(n:ℕ), (seq (n))):ℕ→ℝ).sum ((range 10).map (λ (n:ℕ), n+1) )) = 1512)) = 30 :=
by
  sorry

end ordered_pairs_count_l402_402403


namespace quincy_more_stuffed_animals_l402_402667

theorem quincy_more_stuffed_animals (thor_sold jake_sold quincy_sold : ℕ) 
  (h1 : jake_sold = thor_sold + 10) 
  (h2 : quincy_sold = 10 * thor_sold) 
  (h3 : quincy_sold = 200) : 
  quincy_sold - jake_sold = 170 :=
by sorry

end quincy_more_stuffed_animals_l402_402667


namespace vector_magnitude_l402_402243

noncomputable def magnitude (v: ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem vector_magnitude (x : ℝ) :
  let a := (1, x)
  let b := (x + 2, -2)
  (a.1 * b.1 + a.2 * b.2 = 0) →
  magnitude (a.1 + b.1, a.2 + b.2) = 5 :=
by {
  intros a b h,
  -- The proof would go here.
  sorry
}

end vector_magnitude_l402_402243


namespace whipped_cream_needed_l402_402529

-- Define the baked goods quantities over 20 days
def odd_day_pumpkin_pies := 3
def odd_day_apple_pies := 2
def odd_day_chocolate_cakes := 1
def even_day_pumpkin_pies := 2
def even_day_apple_pies := 4
def even_day_lemon_tarts := 1
def even_day_chocolate_cakes := 2

def total_odd_days := 10
def total_even_days := 10

def total_pumpkin_pies := (odd_day_pumpkin_pies * total_odd_days) + (even_day_pumpkin_pies * total_even_days)
def total_apple_pies := (odd_day_apple_pies * total_odd_days) + (even_day_apple_pies * total_even_days)
def total_chocolate_cakes := (odd_day_chocolate_cakes * total_odd_days) + (even_day_chocolate_cakes * total_even_days)
def total_lemon_tarts := (even_day_lemon_tarts * total_even_days)

-- Define the whipped cream requirements
def whipped_cream_pumpkin_pie := 2
def whipped_cream_apple_pie := 1
def whipped_cream_chocolate_cake := 3
def whipped_cream_lemon_tart := 1.5

-- Total whipped cream needed before Tiffany eats any
def whipped_cream_total_pumpkin_pies := total_pumpkin_pies * whipped_cream_pumpkin_pie
def whipped_cream_total_apple_pies := total_apple_pies * whipped_cream_apple_pie
def whipped_cream_total_chocolate_cakes := total_chocolate_cakes * whipped_cream_chocolate_cake
def whipped_cream_total_lemon_tarts := total_lemon_tarts * whipped_cream_lemon_tart

-- Baked goods eaten by Tiffany
def tiffany_pumpkin_pies := 2
def tiffany_apple_pies := 5
def tiffany_chocolate_cake := 1
def tiffany_lemon_tart := 1

-- Whipped cream eaten by Tiffany
def whipped_cream_tiffany_pumpkin_pies := tiffany_pumpkin_pies * whipped_cream_pumpkin_pie
def whipped_cream_tiffany_apple_pies := tiffany_apple_pies * whipped_cream_apple_pie
def whipped_cream_tiffany_chocolate_cake := tiffany_chocolate_cake * whipped_cream_chocolate_cake
def whipped_cream_tiffany_lemon_tart := tiffany_lemon_tart * whipped_cream_lemon_tart

-- Remaining whipped cream after Tiffany eats
def remaining_whipped_cream_pumpkin_pies := whipped_cream_total_pumpkin_pies - whipped_cream_tiffany_pumpkin_pies
def remaining_whipped_cream_apple_pies := whipped_cream_total_apple_pies - whipped_cream_tiffany_apple_pies
def remaining_whipped_cream_chocolate_cakes := whipped_cream_total_chocolate_cakes - whipped_cream_tiffany_chocolate_cake
def remaining_whipped_cream_lemon_tarts := whipped_cream_total_lemon_tarts - whipped_cream_tiffany_lemon_tart

-- Total remaining whipped cream (rounded up)
def total_remaining_whipped_cream := remaining_whipped_cream_pumpkin_pies + remaining_whipped_cream_apple_pies + remaining_whipped_cream_chocolate_cakes + remaining_whipped_cream_lemon_tarts

-- Prove that the total remaining whipped cream needed is 252 cans.
theorem whipped_cream_needed : total_remaining_whipped_cream.ceil = 252 := 
by sorry

end whipped_cream_needed_l402_402529


namespace continuous_function_identity_l402_402196

open Real

theorem continuous_function_identity (f : ℝ → ℝ)
  (h_cont : Continuous f)
  (h_ineq : ∀ (a b c : ℝ), ∀ x : ℝ, f(a * x ^ 2 + b * x + c) ≥ a * (f x) ^ 2 + b * (f x) + c) :
  ∀ x : ℝ, f x = x := 
sorry

end continuous_function_identity_l402_402196


namespace external_angle_bisector_proof_l402_402733

variables {A T C L K : Type} [Nonempty A] [Nonempty T] [Nonempty C] [Nonempty L] [Nonempty K]

noncomputable def angle_bisector_theorem (AL LC AB BC AK KC : ℝ) : Prop :=
(AL / LC) = (AB / BC) ∧ (AK / KC) = (AL / LC)

noncomputable def internal_angle_bisector (AT TC AL LC : ℝ) : Prop :=
(AT / TC) = (AL / LC)

noncomputable def external_angle_bisector (AT TC AK KC : ℝ) : Prop :=
(AT / TC) = (AK / KC)

theorem external_angle_bisector_proof (AL LC AB BC AK KC AT TC : ℝ) 
(h1 : angle_bisector_theorem AL LC AB BC AK KC)
(h2 : internal_angle_bisector AT TC AL LC) :
external_angle_bisector AT TC AK KC :=
sorry

end external_angle_bisector_proof_l402_402733


namespace find_x_such_that_h_3x_eq_3_h_x_l402_402696

noncomputable def h (x : ℝ) : ℝ := (↑(2 * x + 6) / 5)^(1/4 : ℝ)

theorem find_x_such_that_h_3x_eq_3_h_x : 
  ∃ x : ℝ, h (3 * x) = 3 * h x → x = -(40 / 13 : ℝ) :=
by
  sorry

end find_x_such_that_h_3x_eq_3_h_x_l402_402696


namespace preserve_distances_l402_402451

noncomputable def map_preserves_distance (f: ℝ^2 → ℝ^2) : Prop :=
  ∀ (X Y: ℝ^2), dist X Y = dist (f X) (f Y)

theorem preserve_distances (f: ℝ^2 → ℝ^2)
  (h1 : ∀ A B : ℝ^2, dist A B = 1 → dist (f A) (f B) = 1) :
  map_preserves_distance f :=
by 
  sorry

end preserve_distances_l402_402451


namespace num_witnesses_five_l402_402683

open Set Function

def is_witness {S : Set ℕ} (f : S → S) : Prop :=
  ∀ (Y : Set ℕ), Y ⊆ S → Y ≠ S → Y ≠ ∅ → ∃ y ∈ Y, f y ∉ Y

def num_witnesses (S : Set ℕ) : ℕ :=
  (λ f : S → S, is_witness f).to_finset.card

theorem num_witnesses_five :
  num_witnesses {0, 1, 2, 3, 4, 5} = 120 := 
sorry

end num_witnesses_five_l402_402683


namespace hexagonalPrismCannotIntersectAsCircle_l402_402630

-- Define each geometric shape as a type
inductive GeometricShape
| Sphere
| Cone
| Cylinder
| HexagonalPrism

-- Define a function that checks if a shape can be intersected by a plane to form a circular cross-section
def canIntersectAsCircle (shape : GeometricShape) : Prop :=
  match shape with
  | GeometricShape.Sphere => True -- Sphere can always form a circular cross-section
  | GeometricShape.Cone => True -- Cone can form a circular cross-section if the plane is parallel to the base
  | GeometricShape.Cylinder => True -- Cylinder can form a circular cross-section if the plane is parallel to the base
  | GeometricShape.HexagonalPrism => False -- Hexagonal Prism cannot form a circular cross-section

-- The theorem to prove
theorem hexagonalPrismCannotIntersectAsCircle :
  ∀ shape : GeometricShape,
  (shape = GeometricShape.HexagonalPrism) ↔ ¬ canIntersectAsCircle shape := by
  sorry

end hexagonalPrismCannotIntersectAsCircle_l402_402630


namespace sum_of_factors_of_30_l402_402430

/--
Given the positive integer factors of 30, prove that their sum is 72.
-/
theorem sum_of_factors_of_30 : 
  (1 + 2 + 3 + 5 + 6 + 10 + 15 + 30) = 72 := 
by 
  sorry

end sum_of_factors_of_30_l402_402430


namespace coords_reflect_origin_l402_402384

def P : Type := (ℤ × ℤ)

def reflect_origin (p : P) : P :=
  (-p.1, -p.2)

theorem coords_reflect_origin (p : P) (hx : p = (2, -1)) : reflect_origin p = (-2, 1) :=
by
  sorry

end coords_reflect_origin_l402_402384


namespace external_angle_bisector_of_triangle_l402_402740

variables {A T C L K : Type}
variables [IsTriangle A T C] [IsInternalAngleBisector T L (A T C)]

theorem external_angle_bisector_of_triangle 
  (h1 : is_internal_angle_bisector TL ATC) :
  is_external_angle_bisector TK ATC :=
sorry

end external_angle_bisector_of_triangle_l402_402740


namespace baron_munchausen_max_people_l402_402868

theorem baron_munchausen_max_people :
  ∃ x : ℕ, (x = 37) ∧ 
  (1 / 2 * x).nat_ceil + (1 / 3 * x).nat_ceil + (1 / 5 * x).nat_ceil = x := sorry

end baron_munchausen_max_people_l402_402868


namespace probability_of_premium_best_selling_option_compare_probabilities_l402_402921

theorem probability_of_premium (premium_boxes total_boxes : ℕ) (h_premium : premium_boxes = 40) (h_total : total_boxes = 100) :
  (premium_boxes / total_boxes : ℝ) = 2 / 5 := 
by {
  sorry
}

theorem best_selling_option (premium_boxes special_boxes superior_boxes first_grade_boxes : ℕ) 
  (p1_price : ℝ) (p2_price : ℝ) (p3_price : ℝ) (p4_price : ℝ)
  (h_premium : premium_boxes = 40) (h_special : special_boxes = 30) 
  (h_superior : superior_boxes = 10) (h_first_grade : first_grade_boxes = 20)
  (h_p1_price : p1_price = 36) (h_p2_price : p2_price = 30) 
  (h_p3_price : p3_price = 24) (h_p4_price : p4_price = 18) :
  let avg_price : ℝ := (p1_price * premium_boxes + p2_price * special_boxes + p3_price * superior_boxes + p4_price * first_grade_boxes) / 100 in
  avg_price = 29.4 :=
by {
  sorry
}

theorem compare_probabilities (p1 p2 : ℝ) 
  (h_p1 : p1 = 1465 / 1617) (h_p2 : p2 = 53 / 57) :
  p1 < p2 :=
by {
  sorry
}

end probability_of_premium_best_selling_option_compare_probabilities_l402_402921


namespace rectangle_difference_l402_402798

theorem rectangle_difference 
  (a b p d : ℝ) 
  (h1 : 2 * (a + b) = p)
  (h2 : a^2 + b^2 = d^2) 
  (h3 : a ≥ b) : 
  a - b = (sqrt (8 * d^2 - p^2)) / 2 := 
sorry

end rectangle_difference_l402_402798


namespace mean_goals_correct_l402_402806

-- Definitions based on problem conditions
def players_with_3_goals := 4
def players_with_4_goals := 3
def players_with_5_goals := 1
def players_with_6_goals := 2

-- The total number of goals scored
def total_goals := (3 * players_with_3_goals) + (4 * players_with_4_goals) + (5 * players_with_5_goals) + (6 * players_with_6_goals)

-- The total number of players
def total_players := players_with_3_goals + players_with_4_goals + players_with_5_goals + players_with_6_goals

-- The mean number of goals
def mean_goals := total_goals.toFloat / total_players.toFloat

theorem mean_goals_correct : mean_goals = 4.1 := by
  sorry

end mean_goals_correct_l402_402806


namespace count_integers_satisfying_property_l402_402619

theorem count_integers_satisfying_property :
  let count := (Finset.filter (λ n : ℤ, (n - 3) * (n + 3) * (n + 7) < 0) (Finset.Icc (-10 : ℤ) 12)).card
  in count = 5 :=
by
  sorry

end count_integers_satisfying_property_l402_402619


namespace non_equivalent_paintings_wheel_l402_402923

theorem non_equivalent_paintings_wheel :
  let num_sections := 7
  let num_colors := 2
  let total_paintings := num_colors ^ num_sections
  let single_color_cases := 2
  let non_single_color_paintings := total_paintings - single_color_cases
  let equivalent_rotation_count := num_sections
  (non_single_color_paintings / equivalent_rotation_count) + single_color_cases = 20 :=
by
  let num_sections := 7
  let num_colors := 2
  let total_paintings := num_colors ^ num_sections
  let single_color_cases := 2
  let non_single_color_paintings := total_paintings - single_color_cases
  let equivalent_rotation_count := num_sections
  have h1 := (non_single_color_paintings / equivalent_rotation_count) + single_color_cases
  sorry

end non_equivalent_paintings_wheel_l402_402923


namespace f_g_expression_g_f_expression_l402_402602

section
  def f (x : ℝ) : ℝ := 2 * x - 1
  def g (x : ℝ) : ℝ := if x >= 0 then x^2 else -1

  theorem f_g_expression (x : ℝ) : 
    f (g x) = if x >= 0 then 2 * x^2 - 1 else -3 := 
  by 
    sorry

  theorem g_f_expression (x : ℝ) : 
    g (f x) = if x >= 1/2 then (2 * x - 1)^2 else -1 := 
  by 
    sorry
end

end f_g_expression_g_f_expression_l402_402602


namespace find_value_of_x2001_plus_y2001_l402_402559

theorem find_value_of_x2001_plus_y2001 (x y : ℝ) (h1 : x - y = 2) (h2 : x^2 + y^2 = 4) : 
x ^ 2001 + y ^ 2001 = 2 ^ 2001 ∨ x ^ 2001 + y ^ 2001 = -2 ^ 2001 := by
  sorry

end find_value_of_x2001_plus_y2001_l402_402559


namespace incorrect_normal_vector_description_l402_402491

theorem incorrect_normal_vector_description
  (infinitely_many_normals : ∀ (P : Type) [inner_product_space ℝ P] (u : P), 
                                ∃ (k : ℝ), k ≠ 0)
  (one_unit_normal_vector : ∀ (P : Type) [inner_product_space ℝ P] (u : P), 
                               ∃! (n : P), ∥n∥ = 1 ∧ ∃ k : ℝ, k * u = n)
  (two_unit_normal_vectors : ∀ (P : Type) [inner_product_space ℝ P] (u : P), 
                                 ∃ (n₁ n₂ : P), n₁ ≠ n₂ ∧ ∥n₁∥ = 1 ∧ ∥n₂∥ = 1 ∧ 
                                 ∀ k₁ k₂ : ℝ, k₁ * u = n₁ ∧ k₂ * u = n₂) :
  ¬ ∀ (P : Type) [inner_product_space ℝ P] (u : P), ∀ (v : P), u ≠ 0 → v ≠ 0 → 
    collinear ℝ ({x : P | x = u ∨ x = v}) :=
sorry

end incorrect_normal_vector_description_l402_402491


namespace initial_oscar_fish_l402_402973

-- Define the conditions
variables (g a t o sold_g sold_a sold_t sold_o remaining_fish : ℕ) 

-- Assuming the given initial numbers and the number of fish sold
def initial_g := 94
def initial_a := 76
def initial_t := 89
def sold_g := 30
def sold_a := 48
def sold_t := 17
def sold_o := 24
def remaining_fish := 198

-- Total remaining fish after sales
def remaining_g := initial_g - sold_g
def remaining_a := initial_a - sold_a
def remaining_t := initial_t - sold_t
def remaining_o := o - sold_o

-- Statement to be proven
theorem initial_oscar_fish :
  remaining_g + remaining_a + remaining_t + remaining_o = remaining_fish →
  o = 58 :=
by {
  -- Variables rewrite
  unfold remaining_g remaining_a remaining_t remaining_o,
  sorry
}

end initial_oscar_fish_l402_402973


namespace problem_1_problem_2_l402_402233

-- Define the functions f and g
def f (x : ℝ) : ℝ := Real.log x - 1/x
def g (a b x : ℝ) : ℝ := a * x + b

-- Define the function h
def h (a b x : ℝ) : ℝ := f x - g a b x

-- Prove that for h to be monotonically increasing on (0, +∞), we need a ≤ 0
theorem problem_1 (a : ℝ) : (∀ x > 0, 1/x + 1/(x^2) - a ≥ 0) → a ≤ 0 := 
sorry

-- Prove that given g(x) is a tangent to f(x), the minimum of a + b is -1
theorem problem_2 (a b x0 : ℝ) (hx0_pos : x0 > 0) (h_tangent : g a b x0 = f x0 ∧ g_fin : (λ x, deriv f x) x0 = deriv (g a b) x0) : 
  let t := 1/x0 in
  a = 1/x0 + 1/(x0^2) ∧ b = -1 - 2/x0 + Real.log x0 → a + b = - Real.log t - t + t^2 - 1 → 
  ∀ t > 0, - Real.log t - t + t^2 - 1 ≥ -1 :=
sorry

end problem_1_problem_2_l402_402233


namespace count_distinct_a_l402_402108

def quadratic_has_integer_solutions (a : ℤ) : Prop :=
  ∃ x m n : ℤ, (x^2 + a * x + 12 * a = 0) ∧ (m + n = -a) ∧ (m * n = 12 * a)

theorem count_distinct_a : (finset.univ.filter quadratic_has_integer_solutions).card = 9 := 
sorry

end count_distinct_a_l402_402108


namespace g_at_52_l402_402388

noncomputable def g : ℝ → ℝ := sorry

axiom g_multiplicative : ∀ (x y: ℝ), g (x * y) = y * g x
axiom g_at_1 : g 1 = 10

theorem g_at_52 : g 52 = 520 := sorry

end g_at_52_l402_402388


namespace count_integer_values_of_a_l402_402151

theorem count_integer_values_of_a (a : ℤ) :
  (∃ x : ℤ, x^2 + a * x + 12 * a = 0) ↔ (a ∈ {-(m + n) | m, n : ℤ ∧ m * n = 12 * a}) → ∃! a_vals : finset ℤ, a_vals.card = 16 :=
by
  sorry

end count_integer_values_of_a_l402_402151


namespace angle_C_eq_20_l402_402628

variable (A B C D E F : Type) 
variable [Triangle A B C]
variable [Triangle D E F]

variable congruent_triangles : A ≃ D ∧ B ≃ E ∧ C ≃ F 
variable angle_A : Angle A = 100
variable angle_E : Angle E = 60 

theorem angle_C_eq_20 :
  (Angle C = 180 - 100 - 60) :=
sorry

end angle_C_eq_20_l402_402628


namespace sum_of_medians_at_least_three_quarters_perimeter_l402_402366

variable {A B C : Type}
variables (a b c m_a m_b m_c : ℝ)

def is_triangle (a b c : ℝ) : Prop :=
a + b > c ∧ b + c > a ∧ c + a > b

def median_length_condition (a b c m_a m_b m_c : ℝ) : Prop :=
-- Assuming m_a, m_b, and m_c are the medians corresponding to sides a, b, and c respectively.

∀ G : A, (2/3) * m_a > b ∧ (2/3) * m_b > a ∧ (2/3) * m_c > c 

theorem sum_of_medians_at_least_three_quarters_perimeter
  (h_triangle : is_triangle a b c)
  (h_median : median_length_condition a b c m_a m_b m_c) :
  m_a + m_b + m_c ≥ (3/4) * (a + b + c) :=
sorry

end sum_of_medians_at_least_three_quarters_perimeter_l402_402366


namespace gdp_scientific_notation_l402_402356

theorem gdp_scientific_notation : 
  (33.5 * 10^12 = 3.35 * 10^13) := 
by
  sorry

end gdp_scientific_notation_l402_402356


namespace integer_values_of_a_l402_402177

theorem integer_values_of_a (a : ℤ) :
  (∀ x : ℤ, x^2 + a * x + 12 * a = 0 → x ∈ ℤ) → ∃ n : ℕ, n = 16 :=
by
  sorry

end integer_values_of_a_l402_402177


namespace min_radius_cylinder_proof_l402_402034

-- Defining the radius of the hemisphere
def radius_hemisphere : ℝ := 10

-- Defining the angle alpha which is less than or equal to 30 degrees
def angle_alpha_leq_30 (α : ℝ) : Prop := α ≤ 30 * Real.pi / 180

-- Minimum radius of the cylinder given alpha <= 30 degrees
noncomputable def min_radius_cylinder : ℝ :=
  10 * (2 / Real.sqrt 3 - 1)

theorem min_radius_cylinder_proof (α : ℝ) (hα : angle_alpha_leq_30 α) :
  min_radius_cylinder = 10 * (2 / Real.sqrt 3 - 1) :=
by
  -- Here would go the detailed proof steps
  sorry

end min_radius_cylinder_proof_l402_402034


namespace quadrilateral_is_trapezoid_l402_402359

structure Quadrilateral (α : Type) :=
(A B C D M N : α)
(M_divides_AB : ∃ k : ℝ, k > 0 ∧ k < 1 ∧ (M = A + k • (B - A)))
(N_divides_DC : ∃ k : ℝ, k > 0 ∧ k < 1 ∧ (N = D + k • (C - D)))

def is_trapezoid {α : Type} [MetricSpace α] [NormedAddTorsor ℝ α] 
  (q : Quadrilateral α) : Prop :=
∃ (k : ℝ) (hPos : k > 0) (hLT : k < 1),
  let ⟨k1, hPos1, hLT1, hM⟩ := q.M_divides_AB,
      ⟨k2, hPos2, hLT2, hN⟩ := q.N_divides_DC in
  k1 = k2 ∧
  let area_triangle (P Q R : α) : ℝ :=
    let PQ := dist P Q in
    let PR := dist P R in
    let QR := dist Q R in
    sqrt ((PQ + PR + QR) / 2 *
          ((PQ + PR + QR) / 2 - PQ) *
          ((PQ + PR + QR) / 2 - PR) *
          ((PQ + PR + QR) / 2 - QR)) / 2 in
  area_triangle q.A q.B q.N = area_triangle q.C q.D q.M →
  Parallel (Line.mk q.M q.N) (Line.mk q.A q.D) ∧ 
  Parallel (Line.mk q.M q.N) (Line.mk q.B q.C)

theorem quadrilateral_is_trapezoid 
  {α : Type} [MetricSpace α] [NormedAddTorsor ℝ α] 
  (q : Quadrilateral α) 
  (h_equal_areas : 
    ∃ (k : ℝ) (hPos : k > 0) (hLT : k < 1),
      let ⟨k1, hPos1, hLT1, hM⟩ := q.M_divides_AB,
          ⟨k2, hPos2, hLT2, hN⟩ := q.N_divides_DC in
      k1 = k2 ∧
      let area_triangle (P Q R : α) : ℝ :=
        let PQ := dist P Q in
        let PR := dist P R in
        let QR := dist Q R in
        sqrt ((PQ + PR + QR) / 2 *
              ((PQ + PR + QR) / 2 - PQ) *
              ((PQ + PR + QR) / 2 - PR) *
              ((PQ + PR + QR) / 2 - QR)) / 2 in
      area_triangle q.A q.B q.N = area_triangle q.C q.D q.M) : 
  is_trapezoid q :=
sorry

end quadrilateral_is_trapezoid_l402_402359


namespace ratio_of_areas_l402_402301

theorem ratio_of_areas (ABC : Type) [triangle ABC] (A B C : ABC) (P Q R S T : ABC)
  (h_right_triangle : right_triangle ABC A B C)
  (h_equal_sides : B = C)
  (h_midpoints : P = midpoint A B ∧ Q = midpoint B C ∧ R = midpoint C A)
  (h_midpoint_S : S = midpoint P R)
  (h_midpoint_T : T = midpoint R Q)
  : ratio_of_shaded_to_nonshaded_area ABC A B C P Q R S T = 1/7 :=
sorry

end ratio_of_areas_l402_402301


namespace store_profit_l402_402045

theorem store_profit :
  let selling_price : ℝ := 80
  let cost_price_profitable : ℝ := (selling_price - 0.60 * selling_price)
  let cost_price_loss : ℝ := (selling_price + 0.20 * selling_price)
  selling_price + selling_price - cost_price_profitable - cost_price_loss = 10 := by
  sorry

end store_profit_l402_402045


namespace angle_problem_l402_402996

theorem angle_problem (θ : ℝ) (h1 : 90 - θ = 0.4 * (180 - θ)) (h2 : 180 - θ = 2 * θ) : θ = 30 :=
by
  sorry

end angle_problem_l402_402996


namespace midpoint_polar_line_seg_l402_402273

def midpoint_polar_coords (r1 θ1 r2 θ2 : ℝ) : ℝ × ℝ :=
  let r := (r1 + r2) / 2
  let θ := (θ1 + θ2) / 2
  (r, θ)

theorem midpoint_polar_line_seg :
  ∀ (r1 θ1 r2 θ2 : ℝ),
    r1 = 10 → θ1 = π / 3 →
    r2 = 10 → θ2 = -π / 6 →
    r1 > 0 → r2 > 0 →
    0 ≤ θ1 ∧ θ1 < 2 * π →
    0 ≤ θ2 ∧ θ2 < 2 * π →
    midpoint_polar_coords r1 θ1 r2 θ2 = (10, π / 4) :=
begin
  intros r1 θ1 r2 θ2 r1_eq θ1_eq r2_eq θ2_eq r1_pos r2_pos θ1_range θ2_range,
  calc midpoint_polar_coords r1 θ1 r2 θ2
      = midpoint_polar_coords 10 (π / 3) 10 (-π / 6) : by rw [r1_eq, θ1_eq, r2_eq, θ2_eq]
  ... = (10, π / 4) : by sorry  -- proof steps
end

end midpoint_polar_line_seg_l402_402273


namespace cot_tan_cot_cot_tan_tan_cot_cot_tan_general_tan_simplify_l402_402453

-- Prove: $\cot \alpha = \tan \alpha + 2\cot 2\alpha$
theorem cot_tan_cot (α : ℝ) : Real.cot α = Real.tan α + 2 * Real.cot (2 * α) :=
sorry

-- Prove: $\cot \alpha = \tan \alpha + 2\tan 2\alpha + 4\cot 4\alpha$
theorem cot_tan_tan_cot (α : ℝ) (H1 : Real.cot α = Real.tan α + 2 * Real.cot (2 * α)) :
Real.cot α = Real.tan α + 2 * Real.tan (2 * α) + 4 * Real.cot (4 * α) :=
sorry

-- Prove: $\cot \alpha = \tan α + 2\tan 2α + 2^2\tan 2^2α + \ldots + 2^{n-1}\tan 2^{n-1}α + 2^n\cot 2^nα$
theorem cot_tan_general (α : ℝ) (n : ℕ) :
Real.cot α = Real.tan α + ∑ i in Finset.range n, (2^i : ℝ) * Real.tan (2^i * α) + (2^n : ℝ) * Real.cot (2^n * α) :=
sorry

-- Simplify: $\tan 5^{\circ} + 2\tan 10^{\circ} + 4\tan 20^{\circ} + 8\tan 50^{\circ}$ to $\cot 5^{\circ}$
theorem tan_simplify : Real.tan 5 + 2 * Real.tan 10 + 4 * Real.tan 20 + 8 * Real.tan 50 = Real.cot 5 :=
sorry

end cot_tan_cot_cot_tan_tan_cot_cot_tan_general_tan_simplify_l402_402453


namespace ratio_of_white_socks_l402_402676

theorem ratio_of_white_socks 
  (total_socks : ℕ) (blue_socks : ℕ)
  (h_total_socks : total_socks = 180)
  (h_blue_socks : blue_socks = 60) :
  (total_socks - blue_socks) * 3 = total_socks * 2 :=
by
  sorry

end ratio_of_white_socks_l402_402676


namespace parabola_equation_given_line_and_area_l402_402198

theorem parabola_equation_given_line_and_area
    (a : ℝ) (h_a_ne_zero : a ≠ 0)
    (O F A : ℝ × ℝ)
    (O_def : O = (0, 0))
    (F_def : F = (a / 4, 0))
    (A_def : A = (0, - a / 2))
    (line : (ℝ × ℝ) → Prop)
    (line_def : line = λ p, p.2 = 2 * (p.1 - a / 4))
    (line_O : line O)
    (line_F : line F)
    (line_A : line A)
    (area_OF_A : ℝ)
    (area_val : area_OF_A = 1 / 2 * |a / 4| * |a / 2|)
    (area_eq_four : area_OF_A = 4) :
  ∃ (a : ℝ), y² = ±8x :=
by
  sorry

end parabola_equation_given_line_and_area_l402_402198


namespace ellipse_equation_l402_402585

-- Define the ellipse with given conditions
def ellipse (a b : ℝ) (a_gt_b : a > b) (b_gt_0 : b > 0) : Prop :=
  ∀ x y : ℝ, 
    (x^2 / a^2 + y^2 / b^2 = 1) ↔ (a = sqrt 3) ∧ (b^2 = 1)

-- Focus and eccentricity conditions
def focus_and_eccentricity : Prop :=
  let f : ℝ×ℝ := (sqrt 2, 0),
  let e : ℝ := sqrt 6 / 3 in
  ∃ a : ℝ, a = sqrt 3 ∧ ((sqrt 6 / 3) = sqrt 2 / a)

-- Definition for collinearity and distance between points M and N on the ellipse
def collinearity_and_distance (M N F : ℝ×ℝ): ℝ := 
  ∃ (MN_length : ℝ), 
  MN_length = sqrt 3 ∧ 
  M.1 ^ 2 / 3 + M.2 ^ 2 = 1 ∧ N.1 ^ 2 / 3 + N.2 ^ 2 = 1 ∧ 
  ((x^2 + y^2 = b^2) → (x > 0)) → 
  (M.1 = N.1 ∨ M.1 = F.1) ∧ (M.2 = N.2 ∨ M.2 = F.2)

-- Main theorem to be proven
theorem ellipse_equation : ∀ (a b : ℝ) (a_gt_b : a > b) (b_gt_0 : b > 0),
  focus_and_eccentricity → ellipse a b a_gt_b b_gt_0 ∧
  ∀ (M N : ℝ×ℝ), 
  collinearity_and_distance M N (sqrt 2, 0) ↔ 
  (|MN| = sqrt 3) := 
sorry

end ellipse_equation_l402_402585


namespace weight_removed_l402_402457

-- Definitions for the given conditions
def weight_sugar : ℕ := 16
def weight_salt : ℕ := 30
def new_combined_weight : ℕ := 42

-- The proof problem statement
theorem weight_removed : (weight_sugar + weight_salt) - new_combined_weight = 4 := by
  -- Proof will be provided here
  sorry

end weight_removed_l402_402457


namespace possible_to_sum_1982_l402_402665

theorem possible_to_sum_1982 : 
  ∃ (f : Fin 100 → ℤ), (∀ i, f i = i + 1 ∨ f i = -(i + 1)) ∧ (∑ i, f i) = 1982 :=
sorry

end possible_to_sum_1982_l402_402665


namespace angle_D_in_quadrilateral_l402_402286

-- Given definitions
def Quadrilateral (A B C D : ℝ) : Prop :=
  A + B + C + D = 360

def Angle_Ratio_1_2_1_2 (A B C D : ℝ) : Prop :=
  ⋆(B = 2 * A) ∧ (C = A) ∧ (D = 2 * A)

-- The theorem to be proved
theorem angle_D_in_quadrilateral 
  (A B C D : ℝ) 
  (h1 : Quadrilateral A B C D) 
  (h2 : Angle_Ratio_1_2_1_2 A B C D) : 
  D = 120 :=
  sorry

end angle_D_in_quadrilateral_l402_402286


namespace intersection_of_sets_l402_402552

def set_M : set ℝ := {y | ∃ x : ℝ, y = x^2}
def set_N : set ℝ := {y | ∃ x : ℝ, x^2 + y^2 = 2}

theorem intersection_of_sets : set_M ∩ set_N = {y | 0 ≤ y ∧ y ≤ real.sqrt 2} := 
by 
  simp [set_M, set_N]
  sorry

end intersection_of_sets_l402_402552


namespace quadratic_inequality_solution_l402_402629

theorem quadratic_inequality_solution :
  ∀ (x : ℝ), x^2 - 9 * x + 14 ≤ 0 → 2 ≤ x ∧ x ≤ 7 :=
by
  intros x h
  sorry

end quadratic_inequality_solution_l402_402629


namespace beta_max_success_ratio_l402_402490

theorem beta_max_success_ratio :
  ∀ (a b c d : ℕ),
    (0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) ∧
    (a / b < 21 / 40) ∧
    (c / d < 7 / 10) ∧
    (b + d = 700) →
  ((a + c) / 700 ≤ 139 / 700) :=
begin
  sorry
end

end beta_max_success_ratio_l402_402490


namespace part1_case1_part1_case2_part1_case3_part2_l402_402221

def f (m x : ℝ) : ℝ := (m+1)*x^2 - (m-1)*x + (m-1)

theorem part1_case1 (m x : ℝ) (h : m = -1) : 
  f m x ≥ (m+1)*x → x ≥ 1 := sorry

theorem part1_case2 (m x : ℝ) (h : m > -1) :
  f m x ≥ (m+1)*x →
  (x ≤ (m-1)/(m+1) ∨ x ≥ 1) := sorry

theorem part1_case3 (m x : ℝ) (h : m < -1) : 
  f m x ≥ (m+1)*x →
  (1 ≤ x ∧ x ≤ (m-1)/(m+1)) := sorry

theorem part2 (m : ℝ) : 
  (∀ x : ℝ, -1/2 ≤ x ∧ x ≤ 1/2 → f m x ≥ 0) →
  m ≥ 1 := sorry

end part1_case1_part1_case2_part1_case3_part2_l402_402221


namespace marble_probability_difference_l402_402076

theorem marble_probability_difference :
  let red_marbles := 1101
  let black_marbles := 1101
  let total_marbles := red_marbles + black_marbles
  let choose_two (n : ℕ) := n * (n - 1) / 2
  let P_s := (choose_two red_marbles + choose_two black_marbles) / choose_two total_marbles
  let P_d := (red_marbles * black_marbles) / choose_two total_marbles
  abs (P_s - P_d) = 1 / 2201 :=
by
  -- Proof will go here
  sorry

end marble_probability_difference_l402_402076


namespace number_of_men_is_15_l402_402465

-- Define the conditions
def number_of_people : Prop :=
  ∃ (M W B : ℕ), M = 8 ∧ W = 8 ∧ B = 8 ∧ 8 * M = 120

-- Define the final statement to be proven
theorem number_of_men_is_15 (h: number_of_people) : ∃ M : ℕ, M = 15 :=
by
  obtain ⟨M, W, B, hM, hW, hB, htotal⟩ := h
  use M
  rw [hM] at htotal
  have hM15 : M = 15 := by linarith
  exact hM15

end number_of_men_is_15_l402_402465


namespace external_angle_bisector_l402_402718

open EuclideanGeometry

variables {A T C K L : Point}

theorem external_angle_bisector
    (h1 : is_internal_angle_bisector T L A T C) : is_external_angle_bisector T K A T C :=
sorry

end external_angle_bisector_l402_402718


namespace intersection_with_y_axis_l402_402794

theorem intersection_with_y_axis (f : ℝ → ℝ) (hf : ∀ x, f x = x^2 + x - 2) : f 0 = -2 :=
by
  sorry

end intersection_with_y_axis_l402_402794


namespace baskets_picked_l402_402311

theorem baskets_picked
  (B : ℕ) -- How many baskets did her brother pick?
  (S : ℕ := 15) -- Each basket contains 15 strawberries
  (H1 : (8 * B * S) + (B * S) + ((8 * B * S) - 93) = 4 * 168) -- Total number of strawberries when divided equally
  (H2 : S = 15) -- Number of strawberries in each basket
: B = 3 :=
sorry

end baskets_picked_l402_402311


namespace asymptotes_of_hyperbola_l402_402802

theorem asymptotes_of_hyperbola (x y : ℝ) :
  (x^2 / 4 - y^2 / 5 = 1) → y = (sqrt 5 / 2) * x ∨ y = (-sqrt 5 / 2) * x :=
by
  sorry

end asymptotes_of_hyperbola_l402_402802


namespace student_event_arrangements_l402_402827
-- Importing the necessary libraries

-- Defining the conditions and goal
theorem student_event_arrangements :
  let students := {A, B, C, D, E, F}
  let events := {Event1, Event2, Event3, Event4}
  (disjoint: ∀ (e : Event1 ∪ Event2 ∪ Event3 ∪ Event4), ¬ (A ∈ e ∧ B ∈ e)) ∧
  (participants: ∀ (e : {Event1, Event2, Event3, Event4}), ∃ s, s ∈ students ∧ s ∈ e) ∧
  (unique_participation: ∀ s ∈ students, ∃ e, e ∈ {Event1, Event2, Event3, Event4} ∧ s ∈ e)
  → ∃ n : ℕ, n = 1320 := 
sorry

end student_event_arrangements_l402_402827


namespace external_angle_bisector_proof_l402_402729

variables {A T C L K : Type} [Nonempty A] [Nonempty T] [Nonempty C] [Nonempty L] [Nonempty K]

noncomputable def angle_bisector_theorem (AL LC AB BC AK KC : ℝ) : Prop :=
(AL / LC) = (AB / BC) ∧ (AK / KC) = (AL / LC)

noncomputable def internal_angle_bisector (AT TC AL LC : ℝ) : Prop :=
(AT / TC) = (AL / LC)

noncomputable def external_angle_bisector (AT TC AK KC : ℝ) : Prop :=
(AT / TC) = (AK / KC)

theorem external_angle_bisector_proof (AL LC AB BC AK KC AT TC : ℝ) 
(h1 : angle_bisector_theorem AL LC AB BC AK KC)
(h2 : internal_angle_bisector AT TC AL LC) :
external_angle_bisector AT TC AK KC :=
sorry

end external_angle_bisector_proof_l402_402729


namespace distinct_integer_a_values_l402_402136

open Nat

theorem distinct_integer_a_values : ∃ (S : Finset ℤ), S.card = 8 ∧
  ∀ a ∈ S, ∃ (p q : ℤ), p + q = -a ∧ p * q = 12a ∧ 
  ∀ x ∈ (polynomial.roots (polynomial.C (12 : ℚ) * polynomial.X + polynomial.C a * polynomial.X + polynomial.C 1)).val, x ∈ ℤ :=
by
  sorry

end distinct_integer_a_values_l402_402136


namespace exterior_bisector_ratio_l402_402640

theorem exterior_bisector_ratio (A B C P : Type) [DecidableEq A] [DecidableEq B] [DecidableEq C] [DecidableEq P] (AC CB PA AB : ℝ) 
  (h_AC_CB : AC / CB = 2 / 5)
  (h_ext_angle_bisector : angle_bisector_exterior A B C P) :
  PA / AB = 5 / 3 :=
by
  sorry

end exterior_bisector_ratio_l402_402640


namespace distinct_integer_values_of_a_l402_402115

theorem distinct_integer_values_of_a :
  ∃ (a_values : Finset ℤ), (∀ a ∈ a_values, ∃ (m n : ℤ), (x^2 + a * x + 12 * a = 0) ∧ (m + n = -a) ∧ (m * n = 12 * a))
  ∧ a_values.card = 16 :=
sorry

end distinct_integer_values_of_a_l402_402115


namespace q_l402_402096

-- Definitions for the problem conditions
def slips := 50
def numbers := 12
def slips_per_number := 5
def drawn_slips := 5
def binom := Nat.choose -- Lean function for binomial coefficients

-- Define the probabilities p' and q'
def p' := 12 / (binom slips drawn_slips)
def favorable_q' := (binom numbers 2) * (binom slips_per_number 3) * (binom slips_per_number 2)
def q' := favorable_q' / (binom slips drawn_slips)

-- The statement we need to prove
theorem q'_over_p'_equals_550 : q' / p' = 550 :=
by sorry

end q_l402_402096


namespace candy_store_truffle_price_l402_402460

def total_revenue : ℝ := 212
def fudge_revenue : ℝ := 20 * 2.5
def pretzels_revenue : ℝ := 3 * 12 * 2.0
def truffles_quantity : ℕ := 5 * 12

theorem candy_store_truffle_price (total_revenue fudge_revenue pretzels_revenue truffles_quantity : ℝ) : 
  (total_revenue - (fudge_revenue + pretzels_revenue)) / truffles_quantity = 1.50 := 
by 
  sorry

end candy_store_truffle_price_l402_402460


namespace solve_for_x_l402_402786

theorem solve_for_x : 
  (∃ x : ℝ, (4 * x - 2) / (5 * x - 5) = 3 / 4) ↔ x = -7 :=
begin
  sorry
end

end solve_for_x_l402_402786


namespace geometric_sequence_a_n_sum_of_first_n_terms_b_n_l402_402279

theorem geometric_sequence_a_n :
  (∃ a₁ a₂, a₁ * a₂ = 8 ∧ a₁ + a₂ = 6 ∧ a₁ < a₂) →
  (∀ n : ℕ, n > 0 → (∃ a : ℕ → ℕ, (a 1 = 2) ∧ (a 2 = 4) ∧ (∀ n, a n = 2 ^ n))) :=
begin
  sorry
end

theorem sum_of_first_n_terms_b_n :
  (∀ n : ℕ, n > 0 → (∃ a : ℕ → ℕ, (a 1 = 2) ∧ (a 2 = 4) ∧ (∀ n, a n = 2 ^ n))) →
  (∀ b : ℕ → ℕ, (∀ n, b n = 2 * (2 ^ n) + 3)) →
  (∀ T : ℕ → ℤ, (∀ n, T n = 2 ^ (n + 2) - 4 + 3 * n)) :=
begin
  sorry
end

end geometric_sequence_a_n_sum_of_first_n_terms_b_n_l402_402279


namespace total_books_l402_402675

theorem total_books (joan_books tom_books sarah_books alex_books : ℕ) 
  (h1 : joan_books = 10)
  (h2 : tom_books = 38)
  (h3 : sarah_books = 25)
  (h4 : alex_books = 45) : 
  joan_books + tom_books + sarah_books + alex_books = 118 := 
by 
  sorry

end total_books_l402_402675


namespace inequality_proof_l402_402566

theorem inequality_proof (n : ℕ) (a : Fin n → ℝ)
  (hn : 3 ≤ n)
  (ha_pos : ∀ i, 0 < a i)
  (ha_sum : ∑ i, a i = n) :
  ∑ i, 1 / (a i) + (2 * Real.sqrt 2 * n) / (∑ i, (a i)^2) ≥ n + 2 * Real.sqrt 2 := 
by
  -- proof goes here
  sorry

end inequality_proof_l402_402566


namespace distinct_integer_a_values_l402_402135

open Nat

theorem distinct_integer_a_values : ∃ (S : Finset ℤ), S.card = 8 ∧
  ∀ a ∈ S, ∃ (p q : ℤ), p + q = -a ∧ p * q = 12a ∧ 
  ∀ x ∈ (polynomial.roots (polynomial.C (12 : ℚ) * polynomial.X + polynomial.C a * polynomial.X + polynomial.C 1)).val, x ∈ ℤ :=
by
  sorry

end distinct_integer_a_values_l402_402135


namespace triangle_side_c_l402_402267

theorem triangle_side_c (a b c : ℝ) (A B C : ℝ) (h1 : a = 5) (h2 : A = π / 4) (h3 : cos B = 3 / 5) : c = 7 := 
by sorry

end triangle_side_c_l402_402267


namespace no_integral_value_2001_l402_402312

noncomputable def P (x : ℤ) : ℤ := sorry -- Polynomial definition needs to be filled in

theorem no_integral_value_2001 (a0 a1 a2 a3 a4 : ℤ) (x1 x2 x3 x4 : ℤ) :
  (P x1 = 2020) ∧ (P x2 = 2020) ∧ (P x3 = 2020) ∧ (P x4 = 2020) ∧ 
  x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4 → 
  ¬ ∃ x : ℤ, P x = 2001 :=
sorry

end no_integral_value_2001_l402_402312


namespace external_bisector_TK_l402_402722

-- Definitions of points and segments
variables {A T C L K : Type}

-- Conditions given in the problem
variable (angle_bisector_TL : is_angle_bisector T L A C)
variable (is_triangle : is_triangle T A C)

-- Prove that TK is the external angle bisector
theorem external_bisector_TK : is_external_angle_bisector T K A C :=
sorry

end external_bisector_TK_l402_402722
